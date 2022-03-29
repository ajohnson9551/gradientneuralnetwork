package core;

import java.io.Serializable;

public interface Network extends Serializable {

	double[] evaluate(double[] x);
	void serialize(String path);
	Network deserialize(String path);
}