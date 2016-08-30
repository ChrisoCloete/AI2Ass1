package com.main;

import com.neuralnet.NeuralNetwork;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chris on 8/29/16.
 */
public class Main {
    public static void main(String [] args)   {
        NeuralNetwork nn = new NeuralNetwork(3, 2, 1);
        List<Integer> in = new ArrayList<>();
        in.add(1);
        in.add(4);
        in.add(1);
        nn.feedForward(in);
        Double out = nn.getOutputValues().get(0);
        System.out.println("Out:   " + out);
    }
}

//49900798601494284