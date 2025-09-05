


How Transformer and RNN use the positional/sequential info differently:

Tranformer aggregates the info based on all positions globally through the attention layer, and then pass these position-weighted value vectors through a MLP for each position to project to semantic space.

RNN uses the MLP output from prior positions as inputs to the MLP layer of current position.  

<img src="src/1.png" width="400"/>  