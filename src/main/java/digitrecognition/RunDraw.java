package digitrecognition;

import core.network.Network;

import javax.swing.*;

public class RunDraw {

	public static void main(String[] args) {
		Network net = Network.deserialize("networks");

		JFrame frame = new JFrame("Canvas Example");
		frame.add(new DigitDrawing(net));
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setLayout(null);
		frame.setResizable(false);
		frame.setSize(576, 598);
		frame.setVisible(true);
	}
}
