package digitrecognition;

import core.LayeredNetwork;

import javax.swing.*;

public class Run2 {

	public static void main(String[] args) {
		LayeredNetwork net = new LayeredNetwork("networks");

		JFrame frame = new JFrame("Canvas Example");
		frame.add(new DigitDrawing(net));
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setLayout(null);
		frame.setResizable(false);
		frame.setSize(576, 598);
		frame.setVisible(true);
	}
}
