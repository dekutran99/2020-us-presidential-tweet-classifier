# 2020 US Presidential Tweet Classifier

## Summary

A number of binary classifiers which, given a tweet vector, predict whether the tweet was authored by Donald Trump or Joe Biden

## Models

### Random Forest (no max cap on depth and a forest size of 15 trees)

Result:

- Train error: 0.002749613335624678
- Test error: 0.16537867078825347

### KNN (k=3)

Result:

- Train error: 0.07836398006530332
- Test error: 0.13446676970633695

### Continuous Naive Bayes (no hyperparameters)

Result:

- Train error: 0.24849630520708024
- Test error: 0.22720247295208656

### Stacking

Result:

- Train error: 0.001890359168241966
- Test error: 0.14528593508500773
