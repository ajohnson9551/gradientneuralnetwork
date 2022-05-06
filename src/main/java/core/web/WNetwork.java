package core.web;

import java.io.*;

public class WNetwork implements Serializable {
	public final WLayer[] layers;

	public WNetwork(WLayer[] layers) {
		this.layers = layers;
	}

	public void serialize(String path) {
		try {
			FileOutputStream fileOut = new FileOutputStream(path + "/wnetwork.ser");
			ObjectOutputStream out = new ObjectOutputStream(fileOut);
			out.writeObject(this);
			out.close();
			fileOut.close();
		} catch (IOException i) {
			i.printStackTrace();
		}
	}

	public static WNetwork deserialize(String path) {
		WNetwork wnet;
		try {
			FileInputStream fileIn = new FileInputStream(path + "/wnetwork.ser");
			ObjectInputStream in = new ObjectInputStream(fileIn);
			wnet = (WNetwork) in.readObject();
			in.close();
			fileIn.close();
		} catch (IOException | ClassNotFoundException e) {
			e.printStackTrace();
			return null;
		}
		return wnet;
	}
}
