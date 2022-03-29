package core;

public interface Network {

	double[] evaluate(double[] x);
	void serialize(String path);
	void deserialize(String path);
}