import shared.*;
import func.nn.backprop.*;

import java.io.*;
import java.text.*;

public class NeuralNetTrainer implements Runnable {
    private Instance[] instances;

    /**
     * Number of nodes per layer.
     */
    private int inputeLayerNodes = 13, hiddenLayerNodes = 5, outputLayerNodes = 1;
    
    
    private final BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    private int trainingIterations;
    
    private final GradientErrorMeasure measure = new SumOfSquaresError();
    
    private final WeightUpdateRule updateRule = new RPROPUpdateRule();
    
    /**
     * Directory to output the results to
     */
    private final String dataDir = "results/";
    
    /**
     * What to append after the {@link DataSet}'s name to the results file
     */
    private final String extension;
    
    /**
     * The name of the {@link DataSet}. Used for writing results
     */
    private final String setName;
    
    /**
     * The {@link DataSet} to train and evaluate the Neural Net on
     */
    private final DataSet set;
    
    /** 
     * Error bound
     */
    private double epsilon = 1e-6; 
    

    private static final DecimalFormat df = new DecimalFormat("0.000");

    /**
     * Construct a NeuralNetTrainer with the given parameters using the default epsilon of 0.5
     * @param iterations the number of iterations to train over
     * @param set {@link DataSet} to train on
     * @param extension string to append to the results file (used to delineate on set of results from another)
     * @param name of the {@link DataSet}
     */
    public NeuralNetTrainer(int iterations, DataSet set, String extension, String name) {
    	this.trainingIterations = iterations;
    	this.instances = set.getInstances();
    	this.set = set;
    	this.extension = extension;
    	this.setName = name;
    	this.inputeLayerNodes = this.set.get(0).size();
    }
    
    /**
     * Construct a NeuralNetTrainer with the given parameters with a user-given epsilon value
     * @param iterations
     * @param set
     * @param extension
     * @param name
     */
    public NeuralNetTrainer(int iterations, DataSet set, String extension, String name, double epsilon) {
    	this(iterations, set, extension, name);
    	this.epsilon = epsilon;
    }
    
    /**
     * Train the neural network on the {@link DataSet} provided upon construction, then
     * evaluate its accuracy and print the results to a file
     */
    public void run() {		
    	System.out.println("Running");
        BufferedWriter bw = null;
		try {
			bw = new BufferedWriter(new FileWriter(new File(dataDir + "nn_results_" + setName+extension+".txt")));
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
    	double correct = 0, incorrect = 0;

        BackPropagationNetwork network = factory.createClassificationNetwork(
           new int[] { inputeLayerNodes, hiddenLayerNodes, getOutputLayerNodes() });

        ConvergenceTrainer trainer = new ConvergenceTrainer(
               new BatchBackPropagationTrainer(set, network,
            		   measure, updateRule), epsilon, trainingIterations);
        long start = System.nanoTime();
        double err = trainer.train();
        long end = System.nanoTime();
        double trainingTime = (end - start)/Math.pow(10,9);
        start = System.nanoTime();
        for(int j = 0; j < instances.length; j++) {
        	double predicted, actual;
            actual = Double.parseDouble(instances[j].getLabel().toString());
            predicted = Double.parseDouble(network.getOutputValues().toString());
            System.out.println("Predicted: " + predicted +", Actual: " + actual);
            double trash = Math.abs(predicted - actual) < epsilon ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        double testingTime = (end - start)/Math.pow(10,9);
        try {
			bw.write("Correctly classified " + correct + " instances." +
		            "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
		            + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
		            + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n");
			bw.flush();
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}      
        System.out.println("Trainer Completed");
    }

	public int getOutputLayerNodes() {
		return outputLayerNodes;
	}

	public void setOutputLayerNodes(int outputLayerNodes) {
		this.outputLayerNodes = outputLayerNodes;
	}

}
