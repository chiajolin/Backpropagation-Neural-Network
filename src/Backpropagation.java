import java.util.Random;

public class Backpropagation {
	private final int weightSize = 6;
	private final int patternSize = 3; //every training set has 3 data
	private final int elementSize = 3; // every data has 3 element (input1, input2, expected output)
	private final double[][] trainingSets = {{0,0,0}, {0,1,1}, 
										{1,0,1}, {1,1,0}}; //full data sets
	private double[][] trainingData = new double[patternSize][elementSize]; //training pattern of each batch
	private double learningWeight = Main.learningRate;
	private double targetError = Main.targetError;
	private double[] weights = new double[weightSize];
	private double randomLow = -1;
	private double randomHigh = 1;
	
	private Neuron input1Neuron;
	private Neuron input2Neuron;
	private Neuron hidden1Neuron;
	private Neuron hidden2Neuron;
	private Neuron outputNeuron;
	
	private double totalError = 0;
	private int count = 0;
	private double[] totalDeltas = {0, 0, 0, 0, 0, 0}; //accumulated weight delta
	
	public Backpropagation(){
		genWeights();
		run();
	}
	
	private void run(){	
		//print initial weights
		System.out.println();
		System.out.println("(1) initial weights:");
		printWeight();
		System.out.println("======end of initial weights======");
		System.out.println();
		
		//first batch
		count = 1;
		genTrainingData();
		//generate Neuron for each data in data set
		for(int i = 0; i < trainingData.length; i++){
			genNeuron(i);
			totalError = totalError + comTotalError(i);
		}

		//print first-batch error
		System.out.print("(2) first-batch error:");			
		printError();
		
		while(totalError >= targetError){
			count ++;
			//accumulate deltas from different data
			for(int i = 0; i < trainingData.length; i++){
				accumulateDeltas(i);
			}		
			//batch update weights
			for(int i = 0; i < weightSize; i++){
				updateWeights(i);
			}
			//set accumulated deltas to zero
			setDeltasZero();
			//set Error to 0
			totalError = 0;
			
			//generate new training data
			genTrainingData();			
			//generate Neuron for each data in data set
			for(int i = 0; i < trainingData.length; i++){
				genNeuron(i);
				totalError = totalError + comTotalError(i);
			}		
		}
		System.out.println("(3) final weight:");
		printWeight();
		System.out.println("======end of final weights======");
		System.out.println("(4) final error:" + totalError);
		System.out.println("(5) total number of batches run through in the training:" + count);
	}
	
	//generate different data for each training pattern
	private void genTrainingData(){
		Random rm = new Random();
		for(int i = 0; i < patternSize; i++){
			int data = rm.nextInt( 4 - 1 + 1 ) + 1; //1 ~ 4
			for(int j = 0; j < elementSize; j++){
				trainingData[i][j] = trainingSets[data-1][j];
			}
		}
	}
	
	private void printWeight(){
		for(int i = 1; i < weightSize + 1; i++){
			System.out.println("w" + i + ":" + getWeight(i));
		}
	}
	
	private void printError(){
		System.out.println(totalError);
	}
	
	//generate random initial weights
	private void genWeights(){
		double randomValue = 0;
		Random r = new Random();
		//random weight: -1 ~ 1
		for(int i = 0; i < weightSize; i++){
			randomValue = (randomHigh - randomLow) * r.nextDouble() + randomLow;
			weights[i] = randomValue;
		}
	}

	/**
	 * generate Neuron
	 * @param i: which data set
	 */
	public void genNeuron(int i){
		input1Neuron = new Neuron(trainingData[i][0]);
		input2Neuron = new Neuron(trainingData[i][1]);

		hidden1Neuron = new Neuron(input1Neuron.getOutput(), input2Neuron.getOutput(), getWeight(1), getWeight(2));
		hidden2Neuron = new Neuron(input1Neuron.getOutput(), input2Neuron.getOutput(), getWeight(3), getWeight(4));
		
		outputNeuron = new Neuron(hidden1Neuron.getOutput(), hidden2Neuron.getOutput(), getWeight(5), getWeight(6));
	}
	
	/**
	 * get weight 1 ~ 6
	 * @param i: which weight
	 * @return weight
	 */
	public double getWeight(int i){
		return weights[i - 1];
	}

	/**
	 * compute Etotal (use average error here)
	 * @param i: which data set
	 * @return total error
	 */
	public double comTotalError(int i){
		double target = trainingData[i][2];
		double totalError = 0.5*Math.pow(target-outputNeuron.getOutput(),2) / trainingData.length;
		return totalError;
	}

	/**
	 * accumulate delta to do batch update
	 * @param i: which data set
	 */
	public void accumulateDeltas(int i){
		totalDeltas[0] = totalDeltas[0] + comDeltaHiddenLayer(i, trainingData[i][0], getWeight(5), hidden1Neuron);
		totalDeltas[1] = totalDeltas[1] + comDeltaHiddenLayer(i, trainingData[i][1], getWeight(5), hidden1Neuron);
		totalDeltas[2] = totalDeltas[2] + comDeltaHiddenLayer(i, trainingData[i][0], getWeight(6), hidden2Neuron);
		totalDeltas[3] = totalDeltas[3] + comDeltaHiddenLayer(i, trainingData[i][1], getWeight(6), hidden2Neuron);
		totalDeltas[4] = totalDeltas[4] + comDeltaOutputLayer(i, hidden1Neuron);
		totalDeltas[5] = totalDeltas[5] + comDeltaOutputLayer(i, hidden2Neuron);	
	}
	
	public void setDeltasZero(){
		for(int i = 0; i < weightSize; i++){
			totalDeltas[i] = 0;
		}
	}

	/**
	 * update weight
	 * @param i: which weight
	 */
	public void updateWeights(int i){
		weights[i] = weights[i] + totalDeltas[i];
	}
	
	//compute delta weight of output layer
	private double comDeltaOutputLayer(int i, Neuron n){ //w5 -> need h1 neuron, w6 -> need h2 neuron
		double output = outputNeuron.getOutput();
		double target = trainingData[i][2];
		double outputOfHiddenNeuron = n.getOutput();
		double tmpResult = (output - target) * output * (1 - output) * outputOfHiddenNeuron;
		return -1.0 * learningWeight * tmpResult;		
	}
	
	/**
	 * compute delta weight of hidden layer
	 * @param i: which data set we are training
	 * @param input: which input we need in this formula
	 * @param weight: which weight we need in this formula
	 * @param n: which neuron we need in this formula
	 * @return delta weight of hidden layer
	 */
	private double comDeltaHiddenLayer(int i, double input, double weight, Neuron n){
		double output = outputNeuron.getOutput();
		double target = trainingData[i][2];
		double outputOfHiddenNeuron = n.getOutput();
		double tmp1 = (output - target) * output * (1 - output) * weight;
		double tmp2 = outputOfHiddenNeuron * (1 - outputOfHiddenNeuron);
		double tmp3 = input;
		return -1.0 * learningWeight * tmp1 * tmp2 * tmp3;	
	}
	
	public int getCount(){
		return count;
	}
}
