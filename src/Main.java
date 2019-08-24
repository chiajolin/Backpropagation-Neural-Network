import java.util.Scanner;

public class Main {
	static double learningRate;
	static double targetError;
	//private static double totalError;
	//static Backpropagation bp = new Backpropagation();
	public static void main(String[] args){		
		System.out.println("input the learning rate: ");
		Scanner learingRateScanner = new Scanner(System.in);
		learningRate = learingRateScanner.nextDouble();
		
		System.out.println("input the target error: ");
		Scanner targetErrorScanner = new Scanner(System.in);
		targetError = targetErrorScanner.nextDouble();
		
		Backpropagation bp = new Backpropagation();
	}
}
