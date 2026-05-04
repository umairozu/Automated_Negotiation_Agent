import random
from negmas.sao import SAOCallNegotiator, ResponseType, SAOState, SAOResponse
from negmas.outcomes import Outcome
from negmas.preferences import LambdaMultiFun


class OzuNegotiator(SAOCallNegotiator):
    """
    ANL 2026 deceptive negotiation agent.

    Final score = Advantage + Concealing
      Advantage  = utility(agreement) - reserved_value          (get a good deal)
      Concealing = 1 - opponent's Kendall accuracy on your ufun  (confuse their model)

    Architecture:
      1. Boulware aspiration curve  - stay firm, concede only near the deadline
      2. Deceptive bidding          - randomize within utility band + decoy-issue trick
      3. Frequency opponent model   - estimate what the opponent values from their offers
      4. Multi-condition acceptance - aspiration + safety-net deadline clause
    """

    rational_outcomes = tuple()

    # ------------------------------------------------------------------ #
    #  INITIALIZATION                                                       #
    # ------------------------------------------------------------------ #

    def on_preferences_changed(self, changes):
        """
        Called once at the start of each negotiation when the utility function
        is assigned.  We use it to pre-compute everything we need.
        """
        if self.ufun is None:
            return

        # --- Rational outcomes (utility > reserved), sorted best-first ---
        self.rational_outcomes = tuple(
            o
            for _, o in sorted(
                (
                    (self.ufun(o), o)
                    for o in self.nmi.outcome_space.enumerate_or_sample()
                    if self.ufun(o) > self.ufun.reserved_value
                ),
                reverse=True,
            )
        )

        # --- Opponent model state ---
        self._opp_freq: dict[int, dict] = {}  # {issue_idx: {value: count}}
        self._opp_n = 0                        # total opponent offers seen

        # --- Compute which issues matter most to us (for the decoy trick) ---
        self._issue_imp = self._compute_issue_importance()

        # Decoy setup: pick our LEAST important issue and lock it to one value.
        # The opponent's frequency model will see high consistency on that issue
        # and wrongly conclude we care a lot about it → Concealing score goes up.
        self._decoy_issue: int | None = None
        self._decoy_val = None
        if len(self._issue_imp) > 1 and self.rational_outcomes:
            self._decoy_issue = min(
                range(len(self._issue_imp)), key=lambda i: self._issue_imp[i]
            )
            # Use the most common value for that issue in our rational outcomes
            # so we have plenty of candidates to pick from when filtering.
            from collections import Counter
            vals = [o[self._decoy_issue] for o in self.rational_outcomes]
            self._decoy_val = Counter(vals).most_common(1)[0][0]

        # Flat constant opponent model until we have real data
        self.private_info["opponent_ufun"] = LambdaMultiFun(f=lambda x: 0.5)

    def _compute_issue_importance(self) -> list[float]:
        """
        For each issue, measure the utility RANGE when we vary that issue alone
        (holding all others fixed at the best outcome's values).

        High range  → we care a lot about this issue (high weight in our ufun).
        Low range   → almost irrelevant → good candidate for the decoy.

        This works for any ufun type without needing access to internal weights.
        """
        if not self.rational_outcomes:
            return []

        n_issues = len(self.rational_outcomes[0])
        baseline = list(self.rational_outcomes[0])  # best outcome as probe anchor
        sample = list(self.nmi.outcome_space.enumerate_or_sample())[:300]

        importances = []
        for i in range(n_issues):
            values_i = list({o[i] for o in sample})
            utils_i = []
            for v in values_i:
                probe = baseline[:]
                probe[i] = v
                utils_i.append(float(self.ufun(tuple(probe))))
            importances.append(max(utils_i) - min(utils_i) if utils_i else 0.0)

        return importances

    # ------------------------------------------------------------------ #
    #  MAIN ENTRY POINT                                                    #
    # ------------------------------------------------------------------ #

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        offer = state.current_offer

        if offer is None:
            # First move: we start the negotiation
            return SAOResponse(ResponseType.REJECT_OFFER, self.concealing_bidding_strategy(state))

        # Update our opponent model with the new offer before deciding
        self.update_opponent_model(state)

        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        return SAOResponse(ResponseType.REJECT_OFFER, self.concealing_bidding_strategy(state))

    # ------------------------------------------------------------------ #
    #  ASPIRATION CURVE                                                    #
    # ------------------------------------------------------------------ #

    def _aspiration(self, t: float) -> float:
        """
        Boulware concession curve: aspiration(t) = reserved + (1-reserved)*(1 - t^(1/e))

        With e=0.2  (1/e = 5):
          t=0.50 → asp ≈ 0.969   (still almost at top)
          t=0.70 → asp ≈ 0.832
          t=0.80 → asp ≈ 0.672
          t=0.90 → asp ≈ 0.410
          t=0.95 → asp ≈ 0.226   (near reserved, ready to close)

        Smaller e = more stubborn.  Tune this first if results are bad.
        """
        reserved = float(self.ufun.reserved_value)
        e = 0.2
        return max(reserved, reserved + (1.0 - reserved) * (1.0 - t ** (1.0 / e)))

    # ------------------------------------------------------------------ #
    #  ACCEPTANCE STRATEGY                                                 #
    # ------------------------------------------------------------------ #

    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        Accept when any of these conditions holds:

        1. AC_Const (high bar): offer utility ≥ 90 % of our max utility.
           Never refuse a near-perfect deal regardless of time.

        2. AC_Aspiration: offer utility ≥ current aspiration level.
           The aspiration falls over time (Boulware), so we naturally become
           more willing to accept as the deadline approaches.

        3. AC_Time (deadline safety net): last 5 % of time, accept any
           rational offer.  Prevents us from walking away empty-handed.
        """
        assert self.ufun
        offer = state.current_offer
        if offer is None:
            return False

        my_util = float(self.ufun(offer))
        t = state.relative_time
        reserved = float(self.ufun.reserved_value)

        if my_util <= reserved:
            return False

        # 1. Immediately accept excellent offers
        if my_util >= float(self.ufun.max()) * 0.9:
            return True

        # 2. Accept if offer meets our current aspiration
        if my_util >= self._aspiration(t):
            return True

        # 3. Deadline safety net
        if t > 0.95:
            return True

        return False

    # ------------------------------------------------------------------ #
    #  CONCEALING BIDDING STRATEGY                                         #
    # ------------------------------------------------------------------ #

    def concealing_bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        Three-phase deceptive bidding.

        Core idea: frequency-based opponent models identify your weights by
        watching which issue values you keep constant and which you vary.
        Counter-measure: within the set of outcomes you are willing to accept,
        choose RANDOMLY rather than always picking the same "best" combination.
        Over many rounds the opponent sees many different issue-value combos
        at similar utility levels → they cannot infer your true weights.

        Phase 1  (0 – 40 %): Maximum deception.
            Sample uniformly from ALL outcomes above aspiration.
            Apply decoy filter: only show outcomes where the decoy issue
            is fixed to its fake value → creates one strong false signal.

        Phase 2  (40 – 75 %): Moderate deception.
            Narrow to top 40 % of acceptable outcomes (higher utility floor).
            Still random, still applying decoy filter.

        Phase 3  (75 – 100 %): Convergence.
            Top 15 % only.  Residual randomness still provides some concealing.
            Decoy filter dropped so we don't artificially constrain our choices.
        """
        if not self.rational_outcomes:
            return None

        t = state.relative_time
        asp = self._aspiration(t)

        # All outcomes we are willing to accept (above aspiration)
        acceptable = [o for o in self.rational_outcomes if float(self.ufun(o)) >= asp]
        if not acceptable:
            # Aspiration is too high for any rational outcome — fall back to best 10 %
            n = max(1, len(self.rational_outcomes) // 10)
            acceptable = list(self.rational_outcomes[:n])

        if t < 0.4:
            pool = self._apply_decoy(acceptable)
            return self._opponent_aware_choice(pool)

        elif t < 0.75:
            n = max(1, int(len(acceptable) * 0.4))
            pool = self._apply_decoy(acceptable[:n])
            return self._opponent_aware_choice(pool)

        else:
            n = max(1, int(len(acceptable) * 0.15))
            return self._opponent_aware_choice(acceptable[:n])

    def _opponent_aware_choice(self, candidates: list) -> Outcome | None:
        """
        Choose from the existing candidate pool using the current opponent model.

        This is the only link added between bidding and opponent modeling:
        - before we have enough opponent offers, keep the old random behavior;
        - after the frequency model is available, rank candidates by estimated
          opponent utility and randomly choose among the top few.

        The top-k random choice preserves some concealment/randomness instead of
        always revealing a deterministic opponent-friendly pattern.  To avoid
        expensive scoring in very large domains, the scored set is capped.
        """
        if not candidates:
            return None

        # Keep original behavior while the frequency model is still too noisy.
        if self._opp_n < 3:
            return random.choice(candidates)

        opponent_ufun = self.private_info.get("opponent_ufun", None)
        if opponent_ufun is None:
            return random.choice(candidates)

        # Avoid performance degradation in large candidate pools.
        max_scored = 250
        scored_candidates = (
            random.sample(candidates, max_scored)
            if len(candidates) > max_scored
            else candidates
        )

        try:
            scored = [
                (float(opponent_ufun(o)), random.random(), o)
                for o in scored_candidates
            ]
        except Exception:
            # Safe fallback: never let the opponent model break bidding.
            return random.choice(candidates)

        scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
        top_k = min(5, len(scored))
        return random.choice([o for _, _, o in scored[:top_k]])

    def _apply_decoy(self, candidates: list) -> list:
        """
        Filter the candidate list to outcomes where the decoy issue equals
        our chosen decoy value.

        Why this works: the opponent's frequency model tracks which values
        appear in our offers for each issue.  High consistency (one value
        dominates) is interpreted as "this issue is important to the agent."
        By always showing the same value on our LEAST IMPORTANT issue we
        plant a false belief that it is our highest-priority issue.

        Falls back to the unfiltered list if no candidates match (safe guard).
        """
        if self._decoy_issue is None or self._decoy_val is None:
            return candidates
        filtered = [o for o in candidates if o[self._decoy_issue] == self._decoy_val]
        return filtered if filtered else candidates

    # ------------------------------------------------------------------ #
    #  OPPONENT MODEL                                                      #
    # ------------------------------------------------------------------ #

    def update_opponent_model(self, state: SAOState) -> None:
        """
        Frequency-based opponent modeling (Baarslag / Smith frequency model).

        For each issue i and each value v, we maintain:
            freq[i][v] = number of times the opponent offered value v on issue i

        From this we derive:

        Issue weight  w_i  = consistency_i
                           = (count of most-common value on issue i) / total offers
            Interpretation: if the opponent rarely changes their value on issue i,
            they care about it a lot (high weight).  High variance → low weight.

        Value utility u_i(v) = freq[i][v] / Σ_v freq[i][v]
            Interpretation: how often the opponent offered this value on this issue.
            Frequently offered values are assumed to be preferred by the opponent.

        Estimated opponent utility for an outcome =
            Σ_i  (w_i / Σ_j w_j) * u_i(outcome[i])

        We need at least 3 offers before building a model (too noisy otherwise).
        """
        assert self.ufun
        offer = state.current_offer
        if offer is None:
            return

        self._opp_n += 1
        for i, val in enumerate(offer):
            if i not in self._opp_freq:
                self._opp_freq[i] = {}
            self._opp_freq[i][val] = self._opp_freq[i].get(val, 0) + 1

        if self._opp_n < 3:
            return  # not enough data yet

        # Snapshot current counts into a plain dict so the closure captures
        # frozen data (avoids late-binding bugs when dicts are mutated later).
        freq = {i: dict(vc) for i, vc in self._opp_freq.items()}
        n_issues = len(self.rational_outcomes[0]) if self.rational_outcomes else len(offer)

        def opp_util(outcome) -> float:
            if outcome is None:
                return 0.0

            # Issue weights (consistency)
            weights = []
            for i in range(n_issues):
                if i in freq and freq[i]:
                    total = sum(freq[i].values())
                    top = max(freq[i].values())
                    weights.append(top / total)
                else:
                    weights.append(0.5)

            w_total = sum(weights) or 1.0

            # Weighted sum of value utilities
            util = 0.0
            for i, val in enumerate(outcome):
                w = weights[i] / w_total
                if i in freq and freq[i]:
                    total = sum(freq[i].values())
                    util += w * (freq[i].get(val, 0) / total)
                else:
                    util += w * 0.5
            return util

        self.private_info["opponent_ufun"] = LambdaMultiFun(f=opp_util)
