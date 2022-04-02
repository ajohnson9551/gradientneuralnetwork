package digitrecognition;

import core.network.Network;
import core.Utility;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

public class DigitDrawing extends JPanel implements MouseListener, KeyListener, MouseMotionListener {

	Network net;

	double[] cells = new double[28 * 28];
	int lastX = -1;
	int lastY = -1;

	public DigitDrawing(Network net) {
		this.setBackground (Color.WHITE);
		this.setSize(560, 560);
		this.addMouseListener(this);
		this.addMouseMotionListener(this);
		this.addKeyListener(this);
		this.setFocusable(true);
		this.net = net;
	}

	@Override
	public void paintComponent(Graphics g) {
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				g.setColor(colorFromDouble(cells[i + 28 * j]));
				g.fillRect(20 * i, 20 * j, 20, 20);
			}
		}
	}

	public void resetCells() {
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				cells[i + 28 * j] = 0;
			}
		}
		this.repaint();
	}

	public Color colorFromDouble(double x) {
		return new Color((int) (255 * (1 - x)), (int) (255 * (1 - x)), (int) (255 * (1 - x)));
	}

	public void paintCells(int x, int y) {
		int cellX = toCellIndex(x);
		int cellY = toCellIndex(y);
		increaseToMax(cellX, cellY, 0.5);
		if (cellX != lastX || cellY != lastY) {
			increaseToMax(cellX + 1, cellY, 0.3);
			increaseToMax(cellX - 1, cellY, 0.3);
			increaseToMax(cellX, cellY + 1, 0.3);
			increaseToMax(cellX, cellY - 1, 0.3);
			increaseToMax(cellX + 1, cellY + 1, 0.1);
			increaseToMax(cellX - 1, cellY + 1, 0.1);
			increaseToMax(cellX + 1, cellY + 1, 0.1);
			increaseToMax(cellX - 1, cellY - 1, 0.1);
			this.repaint();
		}
		lastX = cellX;
		lastY = cellY;
	}

	public int toCellIndex(int z) {
		z = z / 20;
		if (z < 0) {
			z = 0;
		} else if (z > 27) {
			z = 27;
		}
		return z;
	}

	public void increaseToMax(int cellX, int cellY, double add) {
		if (cellX >= 0 && cellX < 28 && cellY >= 0 && cellY < 28) {
			cells[cellX + 28 * cellY] = cells[cellX + 28 * cellY] + add < 1 ? cells[cellX + 28 * cellY] + add : 1;
		}
	}

	@Override
	public void mouseClicked(MouseEvent e) {

	}

	@Override
	public void mousePressed(MouseEvent e) {

	}

	@Override
	public void mouseReleased(MouseEvent e) {
		double[] output = net.evaluate(cells, 0);
		System.out.println("I think you drew a " + Utility.maxIndex(output) + "!");
		System.out.print("[ ");
		for (int i = 0; i < 10; i++) {
			System.out.print(i + ": " + Utility.roundString(output[i]) + " ");
		}
		System.out.println("]");
	}

	@Override
	public void mouseEntered(MouseEvent e) {

	}

	@Override
	public void mouseExited(MouseEvent e) {

	}

	@Override
	public void keyTyped(KeyEvent e) {

	}

	@Override
	public void keyPressed(KeyEvent e) {

	}

	@Override
	public void keyReleased(KeyEvent e) {
		if (e.getID() == 402) {
			System.out.println("Resetting!");
			this.resetCells();
		}
	}

	@Override
	public void mouseDragged(MouseEvent e) {
		this.paintCells(e.getX(), e.getY());
	}

	@Override
	public void mouseMoved(MouseEvent e) {

	}
}
