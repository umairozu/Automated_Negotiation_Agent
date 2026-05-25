# OzuNegotiator — ANL 2026 Negotiation Agent

OzuNegotiator is an automated negotiation agent developed for the **ANL 2026 League** of the Automated Negotiating Agents Competition (ANAC).  
The agent is implemented using **NegMAS** and follows the Alternating Offers Protocol.

The main goal of the agent is to reach high-utility agreements while also hiding or misleading the opponent’s model of our real preferences.

## Features

- Two-segment aspiration strategy
- Deceptive early bidding using randomization
- Decoy issue mechanism to confuse opponent models
- Frequency-based opponent modeling
- Opponent-aware late bidding
- Deadline protection using a midpoint counteroffer
- Multi-condition acceptance strategy

## Agent Strategy

The agent divides the negotiation into several phases:

### Early Phase

The agent bids randomly from high-utility acceptable outcomes.  
A decoy issue is used to make the opponent believe that a low-importance issue is important to us.

### Middle Phase

The agent still uses deceptive randomization, but from a smaller and stronger set of candidate bids.

### Late Phase

The agent stops using the decoy and starts using the opponent model.  
It chooses among its own top outcomes and offers the one estimated to be best for the opponent.

### Deadline Phase

Near the deadline, the agent protects itself from cheap last-minute offers.  
If the opponent’s offer is much worse than our previous bid, the agent makes one midpoint counteroffer before accepting any rational final offer.

## Main Components

### Bidding Strategy

The bidding strategy is based on a time-dependent aspiration level.  
Only outcomes above the current aspiration threshold are considered.

### Acceptance Strategy

The agent accepts an offer if:

- The offer is above the reservation value
- The offer is at least 90% of the maximum utility
- The offer satisfies the current aspiration level
- The deadline is very close and the offer is still rational

### Opponent Model

The opponent model is frequency-based.  
It tracks how often the opponent offers each value for each issue.  
Values that appear more often are estimated to be more preferred by the opponent.

### Deception Strategy

The deception strategy uses:

- Randomized bidding inside a safe utility range
- A repeated decoy value on the least important issue
- Delayed opponent-aware bidding to avoid revealing preferences too early

## File Structure

```text
.
├── ozu_negotiator.py   # Main negotiation agent
├── README.md           # Project documentation
└── requirements.txt    # Extra dependencies, if needed
```
## Requirements

* **Python 3.10+**
* **NegMAS** (Negotiation Multi-Agent System)
* **ANL 2026 Skeleton Package**
