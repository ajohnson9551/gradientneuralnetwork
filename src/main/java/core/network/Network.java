package core.network;

import core.web.WNetwork;

import java.io.*;

public abstract class Network implements Serializable {

	public NetworkParameters param;

	Network(NetworkParameters param) {
		this.param = param;
	}

	public abstract double[] evaluate(double[] x, int batchIndex);

	public void serialize(String path) {
		try {
			FileOutputStream fileOut = new FileOutputStream(path + "/network.ser");
			ObjectOutputStream out = new ObjectOutputStream(fileOut);
			out.writeObject(this);
			out.close();
			fileOut.close();
		} catch (IOException i) {
			i.printStackTrace();
		}
	}

	public static Network deserialize(String path) {
		Network net;
		try {
			FileInputStream fileIn = new FileInputStream(path + "/network.ser");
			ObjectInputStream in = new ObjectInputStream(fileIn);
			net = (Network) in.readObject();
			in.close();
			fileIn.close();
		} catch (IOException | ClassNotFoundException e) {
			e.printStackTrace();
			return null;
		}
		return net;
	}
}