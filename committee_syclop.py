#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Let's have a look on the confidance of each network and ask, weather choosing 
or adding another trajectory can help? 
With one network?

First step - 
- Run the confusion matrix over 30 random trajectories
- Pick the five best and most distinct networks from the bunch. 
  i.e. the netwroks with the least mistakes AND that mistake on different imgs
- First ensamble - learn entropy or confidance levels. 
  for each test image, run all trajectories and netwroks. Pick a few answers:
      The one that is most confident (register the entropy)
      An avarage of all 
      A WTA avarage
      Register the entropy of the networks that were right
- Compare all the entropy and ensamble methods, pick an entropy treshold
- Run entropy treshold network - for each img in the test data drew one trajectory 
  get the entropy of the network, if it below treshold, continue to next 
  trajectory. 

Ensamble learning - pick 5 trajectories that have the biggest mistakes variance between them,
use each learnes netwrok together as an ensamble

Choose the best - Attach a small cnn to deside which trajectory to decide given
                 a first glimps of the data. 
"""

