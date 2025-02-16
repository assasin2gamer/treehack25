// File: svm_model.h
#ifndef SVM_MODEL_H
#define SVM_MODEL_H

#include <vector>

class SVMModel {
public:
    SVMModel();
    void train(const std::vector<std::vector<double>> &features, const std::vector<int> &labels, int iterations = 1000, double lambda = 0.01);
    int predict(const std::vector<double> &feature) const;
private:
    std::vector<double> weights;
    double bias;
};

#endif
