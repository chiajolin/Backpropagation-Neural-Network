
public class Neuron {
	private final int inputSize = 2; //for each neuron, except for input layer, there are two inputs to compute for the output.
	private double[] input = new double[inputSize];
	private double[] weight = new double[inputSize];
	private double netTotal = 0;
	private double output = 0;
	
	//constructor for initial input neuron
	public Neuron(double input){
		this.output = input;
	}
	
	public Neuron(double input1, double input2, double weight1, double weight2){
		input[0] = input1;
		input[1] = input2;
		weight[0] = weight1;
		weight[1] = weight2;
		computeNet();
		computeOutput();
	}
	
	//compute the total net input 
	private void computeNet(){
		for(int i = 0; i < inputSize; i++){
			netTotal = netTotal + input[i] * weight[i];
		}
	}
	
	//compute the output
	private void computeOutput(){
		output = 1 / (1 + Math.exp(-netTotal));
	}
	
	//get the total net input 
	public double getNetTotal(){
		return netTotal;
	}
	
	//get the output
	public double getOutput(){
		return output;
	}
}
