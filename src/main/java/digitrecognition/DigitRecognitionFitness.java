package digitrecognition;

import core.Fitness;
import core.Network;
import core.Utility;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;

public class DigitRecognitionFitness implements Fitness {

    // if false, does test
    private final boolean training;
    private final double percentToDo;
    private double[][] images;
    private double[] labels;
    private double[][] answers;
    private final String imagesPathString;
    private final String labelsPathString;

    public DigitRecognitionFitness(boolean training, double percentToDo, boolean onOffPixels) {
        this.training = training;
        this.percentToDo = percentToDo;
        if (training) {
            imagesPathString = "training/train-images.idx3-ubyte";
            labelsPathString = "training/train-labels.idx1-ubyte";
        } else {
            imagesPathString = "testing/t10k-images.idx3-ubyte";
            labelsPathString = "testing/t10k-labels.idx1-ubyte";
        }
        Path imagesPath = Paths.get(".").resolve(imagesPathString);
        Path labelsPath = Paths.get(".").resolve(labelsPathString);

        try {
            DataInputStream imagesDataInputStream =
                    new DataInputStream(new FileInputStream(imagesPath.toFile()));
            DataInputStream labelsDataInputStream =
                    new DataInputStream(new FileInputStream(labelsPath.toFile()));
            int magicImages = imagesDataInputStream.readInt();
            if (magicImages != 0x803)
            {
                throw new IOException("Expected magic header of 0x803 "
                        + "for images, but found " + magicImages);
            }

            int magicLabels = labelsDataInputStream.readInt();
            if (magicLabels != 0x801)
            {
                throw new IOException("Expected magic header of 0x801 "
                        + "for labels, but found " + magicLabels);
            }

            int numberOfImages = imagesDataInputStream.readInt();
            int numberOfLabels = labelsDataInputStream.readInt();

            if (numberOfImages != numberOfLabels)
            {
                throw new IOException("Found " + numberOfImages
                        + " images but " + numberOfLabels + " labels");
            }

            int numRows = imagesDataInputStream.readInt();
            int numCols = imagesDataInputStream.readInt();

            images = new double[numberOfImages][numRows * numCols];
            labels = new double[numberOfLabels];
            answers = new double[numberOfLabels][10];

            for (int n = 0; n < numberOfImages; n++)
            {
                labels[n] = labelsDataInputStream.readByte();
                for (int i = 0; i < 10; i++) {
                    answers[n][i] = i == labels[n] ? 1.0 : 0.0;
                }
                images[n] = new double[numRows * numCols];
                byte[] imagesBytes = new byte[numRows * numCols];
                read(imagesDataInputStream, imagesBytes);
                for (int i = 0; i < numRows * numCols; i++) {
                    images[n][i] = onOffPixels ? (imagesBytes[i] == 0 ? 0.0 : 1.0) : convertByte(imagesBytes[i]);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(0);
        }
    }

    public double convertByte(byte b) {
        double r = 0;
        if (b >= 0) {
            r = b;
        } else {
            r = ((double) b) + 256.0;
        }
        return r / 256.0;
    }

    private void read(InputStream inputStream, byte data[])
            throws IOException
    {
        int offset = 0;
        while (true)
        {
            int read = inputStream.read(
                    data, offset, data.length - offset);
            if (read < 0)
            {
                break;
            }
            offset += read;
            if (offset == data.length)
            {
                return;
            }
        }
        throw new IOException("Tried to read " + data.length
                + " bytes, but only found " + offset);
    }

    public double mse(Network net, int k) {
        return Utility.getUtility().mse(net.evaluate(images[k]), (int) labels[k]);
    }

    @Override
    public double mse(Network net) {
        double mse = 0;
        for (int i = 0; i < ((double) labels.length) * percentToDo; i++) {
            mse += mse(net, i);
        }
        mse = mse / (((double) labels.length) * percentToDo);
        return mse;
    }

    @Override
    public double percentCorrect(Network net) {
        double percent = 0;
        double[] response;
        int guess;
        for (int i = 0; i < ((double) labels.length) * percentToDo; i++) {
            response = net.evaluate(images[i]);
            guess = Utility.getUtility().maxIndex(response);
            percent += guess == labels[i] ? 1.0 : 0.0;
        }
        percent = percent / (((double) labels.length) * percentToDo);
        return percent;
    }

    @Override
    public double[][] getAnswers() {
        return answers;
    }

    @Override
    public double[][] getData() {
        return images;
    }
}
