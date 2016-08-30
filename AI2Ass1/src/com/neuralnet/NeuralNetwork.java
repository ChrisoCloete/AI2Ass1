package com.neuralnet;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chris on 8/29/16.
 */
public class NeuralNetwork {

    private int hiddenLayers = 1;
    private int numEpochs = 500;

    private int numInputs;
    private int numOutputs;
    private int numHidden;

    private double learningRate = 0.5;

    private List<Integer> inputValues;
    private List<Double> hiddenValues;
    private List<Double> outputValues;
    private List<Double> outputError;
    private double[][] weightsIH;
    private double[][] weightsHO;


    public NeuralNetwork(int in, int out, int hidden) {
        numInputs = in;
        numOutputs = out;
        numHidden = hidden;
        weightsIH = new double[numInputs+1][numHidden];
        weightsHO = new double[numHidden+1][numOutputs];
        for(int i = 0; i < numInputs + 1; i++) {
            for(int j = 0; j < numHidden; j++) {
                weightsIH[i][j] = Math.random();
            }
        }
        for(int i = 0; i < numHidden + 1; i++) {
            for(int j = 0; j < numOutputs; j++) {
                weightsHO[i][j] = Math.random();
            }
        }
        inputValues = new ArrayList<>();
        outputValues = new ArrayList<>();
        hiddenValues = new ArrayList<>();
        outputError = new ArrayList<>();
        for(int i= 0; i < numInputs; i++) {
            inputValues.add(0);
        }
        for(int i= 0; i < numHidden; i++) {
            hiddenValues.add(0.0);
        }
        for(int i= 0; i < numOutputs; i++) {
            outputValues.add(0.0);
        }
    }

    double sigmoid(double val){
        return 1.0 / (1 + Math.exp(-val));
    }

    public void feedForward(List<Integer> input) {
        inputValues = input;
        double tmp;
        for(int i = 0; i < numHidden; i++) {
            tmp = 0.0;
            for(int j = 0; j < numInputs; j++) {
                tmp = tmp + (inputValues.get(j) * weightsIH[j][i]);
            }
            hiddenValues.add(sigmoid(tmp - weightsIH[numInputs][i]));
        }
        for(int i = 0; i < numOutputs; i++) {
            tmp = 0.0;
            for(int j = 0; j < numHidden; j++) {
                tmp =+ (hiddenValues.get(j) * weightsHO[j][i]);
            }
            outputValues.add(sigmoid(tmp - weightsHO[numHidden][i]));
        }
    }

    public double calcTotalError(List<Double> exp) {
        double sum = 0;
        for(int i = 0; i < exp.size(); i++) {
            double err = exp.get(i) - outputValues.get(i);
            outputError.add(err);
            err = err*err;
            err = err * 0.5;
            sum += err;
        }
        return sum;
    }

    public List<Double> getOutputValues() {
        return outputValues;
    }

    public void backPropegation(List<Double> expected) {
        double[][] newWeightsHO = new double[numInputs+1][numHidden];
        double[][] newWeightsIH = new double[numHidden+1][numOutputs];
        double[] hiddenSigma = new double[numHidden+1];
        for(int i =0; i < numOutputs; i++) {
            double out = outputValues.get(i);
            double exp = expected.get(i);
            double sigma = (-(exp - out)* out*(1-out));
            for(int j = 0; j < numHidden; j++) {
                double weightChange = learningRate * sigma * hiddenValues.get(j);
                newWeightsHO[j][i] = weightsHO[j][i] - weightChange;

                hiddenSigma[j] += sigma * weightsHO[j][i];
            }
            double weightChange = learningRate * sigma;
            newWeightsHO[numHidden][i] = weightsHO[numHidden][i] - weightChange;
        }
        for(int i =0; i < numHidden; i++) {

            double sigma = hiddenSigma[i] * hiddenValues.get(i) * (1 - hiddenValues.get(i));
            for(int j =0; j < numInputs; j++) {
                double weightChange = learningRate * sigma * inputValues.get(j);
                newWeightsIH[j][i] = weightsIH[j][i] - weightChange;
            }
            double weightChange = learningRate * sigma;
            newWeightsIH[numInputs][i] = weightsIH[numInputs][i] - weightChange;
        }

        for(int i = 0; i < numHidden; i++) {
            for(int j = 0; j < numInputs; j++) {
                weightsIH[j][i] = newWeightsIH[j][i];
            }
            weightsIH[numInputs][i] = newWeightsIH[numInputs][i];
        }

        for(int i = 0; i< numOutputs; i++) {
            for(int j = 0; j < numHidden; j++) {
                weightsHO[j][i] = newWeightsHO[j][i];
            }
            weightsHO[numHidden][i] = newWeightsHO[numHidden][i];
        }

    }


    /*for(int i = 0; i < numOutputs; i++) {
            for (int k = 0; k < numHidden; k++) {
                double weightChange = learningRate * err * hiddenValues.get(k);
                weightsHO[k][i] = weightsHO[k][i] - weightChange;

                //regularisation on the output weights
                if (weightsHO[k][i] < -5) {
                    weightsHO[k][i] = -5;
                } else if (weightsHO[k] > 5) {
                    weightsHO[k][i] = 5;
                }
}
        }
                for(int i = 0;i<numHidden;i++) {
        for(int k = 0;k<numInputs;k++) {
        double x = 1 - (hiddenValues.get(i) * hiddenValues.get(i));
        x = x * weightsHO[i] * err * learningRate;
        x = x * inputValues[k];
        double weightChange = x;
        weightsIH[k][i] = weightsIH[k][i] - weightChange;
        }
        }*/

    /*for(int k = 0;k<numHidden;k++) {
            double weightChange = LR_HO * err * hiddenVal[k];
            weightsHO[k] = weightsHO[k] - weightChange;

            //regularisation on the output weights
            if (weightsHO[k] < -5) {
                weightsHO[k] = -5;
            }
            else if (weightsHO[k] > 5) {
                weightsHO[k] = 5;
            }
        }
        for(int i = 0;i<numHidden;i++) {
            for(int k = 0;k<numInputs;k++) {
                double x = 1 - (hiddenVal[i] * hiddenVal[i]);
                x = x * weightsHO[i] * err * LR_IH;
                x = x * trainInputs[patNum][k];
                double weightChange = x;
                weightsIH[k][i] = weightsIH[k][i] - weightChange;
            }
        }*/

    /* for(int i = 0;i <= numEpochs;i++)
        {
            for(int j = 0;j<numPatterns;j++)
            {
                //select a pattern at random
                patNum = Math.random()%numPatterns;

                //calculate the current network output
                //and error for this pattern
                calcNet();

                //change network weights
                WeightChangesHO();
                WeightChangesIH();
            }

            //display the overall network error
            //after each epoch
            calcOverallError();

            printf("epoch = %d RMS Error = %f\n",i,RMSerror);
        }*/

}




