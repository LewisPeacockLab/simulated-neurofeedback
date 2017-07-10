# simulated-neurofeedback
Simulation of decoded neurofeedback learning.

## Getting started

To manually train a classifier:
```
brain = Simfeed2dBrain;
clf = ReinforcementSimfeed2dClassifier;
clf.sampleClassifier(brain);
```

To use the pre-trained classifier:
```
brain = Simfeed2dBrain;
load('reinforcement_2d_classifier');
```

For sample plots from existing data:
```
plotSimfeed2d;
```
