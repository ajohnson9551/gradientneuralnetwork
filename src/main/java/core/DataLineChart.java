package core;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYSplineRenderer;
import org.jfree.chart.ui.ApplicationFrame;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.awt.*;

public class DataLineChart extends ApplicationFrame {

	double[][] data;
	int cycles;
	String data0Title;
	String data1Title;

	public DataLineChart(String title, String chartTitle, double[][] data, int cycles, double yMin, double yMax, String data0Title, String data1Title) {
		super(title);
		this.data = data;
		this.cycles = cycles;
		this.data0Title = data0Title;
		this.data1Title = data1Title;
		Font font = Font.getFont("Arial");

		XYSeriesCollection scoreData = createScoreDataset();

		XYPlot plot = new XYPlot();
		plot.setDataset(0, scoreData);
		plot.setBackgroundPaint(Color.BLACK);


		XYSplineRenderer sr1 = new XYSplineRenderer();
		sr1.setSeriesShapesVisible(0, false);
		sr1.setSeriesShapesVisible(1, false);
		plot.setRenderer(0, sr1);


		NumberAxis range0 = new NumberAxis("Score");
		range0.setRange(yMin, yMax);
		range0.setTickUnit(new NumberTickUnit(0.1));

		NumberAxis domain = new NumberAxis("Cycle");
		domain.setRange(0, cycles);
		domain.setTickUnit(new NumberTickUnit(cycles/10));

		plot.setRangeAxis(0, range0);
		plot.setDomainAxis(domain);

		plot.mapDatasetToRangeAxis(0, 0);

		JFreeChart lineChart = new JFreeChart(chartTitle, font, plot, true);
		lineChart.setBackgroundPaint(Color.GRAY);
		lineChart.getLegend().setBackgroundPaint(Color.GRAY);
		ChartPanel chartPanel = new ChartPanel(lineChart);
		chartPanel.setPreferredSize(new Dimension(600, 400));
		setContentPane(chartPanel);
	}

	private XYSeriesCollection createScoreDataset() {
		XYSeriesCollection dataset = new XYSeriesCollection();
		XYSeries series0 = new XYSeries(data0Title);
		XYSeries series1 = new XYSeries(data1Title);
		for (int i = 1; i <= data[0].length; i++) {
			series0.add(i, data[0][i-1]);
			series1.add(i, data[1][i-1]);
		}
		dataset.addSeries(series0);
		dataset.addSeries(series1);
		return dataset;
	}
}
