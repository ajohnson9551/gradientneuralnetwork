package core.web;

import core.network.ConvolutionalNetwork;
import core.network.Network;

public class Converter {
	public static void main(String[] args) {
		ConvolutionalNetwork net = (ConvolutionalNetwork) Network.deserialize("networks");
		WNetwork wnet = net.webify();
		String path = "networks";
		wnet.serialize(path);
		System.out.println("Convertion finished, wnetwork serialized.");
	}
}
