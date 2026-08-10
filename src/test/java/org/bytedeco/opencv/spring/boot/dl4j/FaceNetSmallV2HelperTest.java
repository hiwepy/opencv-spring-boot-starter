package org.bytedeco.opencv.spring.boot.dl4j;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;

/**
 * Tests for {@link FaceNetSmallV2Helper}.
 * @author <a href="https://github.com/loong10k">Loong Wan</a>
 */
class FaceNetSmallV2HelperTest {

    @TempDir
    File tempDir;

    @BeforeEach
    void resetIndexes() {
        FaceNetSmallV2Model.reluIndex = 1;
        FaceNetSmallV2Model.paddingIndex = 1;
    }

    @Test
    void reluShouldCreateActivationLayer() {
        ActivationLayer layer = FaceNetSmallV2Helper.relu();
        assertThat(layer).isNotNull();
    }

    @Test
    void zeroPaddingShouldCreateLayer() {
        ZeroPaddingLayer layer = FaceNetSmallV2Helper.zeroPadding(2);
        assertThat(layer).isNotNull();
    }

    @Test
    void convolutionShouldCreateLayer() {
        ConvolutionLayer layer = FaceNetSmallV2Helper.convolution(3, 64, 128);
        assertThat(layer).isNotNull();
    }

    @Test
    void convolutionWithStridesShouldCreateLayer() {
        ConvolutionLayer layer = FaceNetSmallV2Helper.convolution(3, 64, 128, 2);
        assertThat(layer).isNotNull();
    }

    @Test
    void batchNormShouldCreateLayer() {
        BatchNormalization layer = FaceNetSmallV2Helper.batchNorm(64);
        assertThat(layer).isNotNull();
    }

    @Test
    void nextReluIdShouldIncrement() {
        FaceNetSmallV2Model.reluIndex = 1;
        String id1 = FaceNetSmallV2Helper.nextReluId();
        String id2 = FaceNetSmallV2Helper.nextReluId();
        assertThat(id1).isEqualTo("relu1");
        assertThat(id2).isEqualTo("relu2");
    }

    @Test
    void nextPaddingIdShouldIncrement() {
        FaceNetSmallV2Model.paddingIndex = 1;
        String id1 = FaceNetSmallV2Helper.nextPaddingId();
        String id2 = FaceNetSmallV2Helper.nextPaddingId();
        assertThat(id1).isEqualTo("padding1");
        assertThat(id2).isEqualTo("padding2");
    }

    @Test
    void lastPaddingIdShouldReturnPreviousId() {
        FaceNetSmallV2Model.paddingIndex = 5;
        assertThat(FaceNetSmallV2Helper.lastPaddingId()).isEqualTo("padding4");
    }

    @Test
    void lastReluIdShouldReturnPreviousId() {
        FaceNetSmallV2Model.reluIndex = 10;
        assertThat(FaceNetSmallV2Helper.lastReluId()).isEqualTo("relu9");
    }

    @Test
    void readWightsValuesShouldParseFile() throws IOException {
        File weightFile = new File(tempDir, "test_weights.csv");
        Files.writeString(weightFile.toPath(), "1.0,2.5,3.7\n4.2,5.1");
        double[] values = FaceNetSmallV2Helper.readWightsValues(weightFile.getAbsolutePath());
        assertThat(values).hasSize(5);
        assertThat(values[0]).isEqualTo(1.0);
        assertThat(values[1]).isEqualTo(2.5);
        assertThat(values[2]).isEqualTo(3.7);
        assertThat(values[3]).isEqualTo(4.2);
        assertThat(values[4]).isEqualTo(5.1);
    }

    @Test
    void readWightsValuesShouldHandleSingleLine() throws IOException {
        File weightFile = new File(tempDir, "single_line.csv");
        Files.writeString(weightFile.toPath(), "1.1,2.2,3.3");
        double[] values = FaceNetSmallV2Helper.readWightsValues(weightFile.getAbsolutePath());
        assertThat(values).hasSize(3);
        assertThat(values[0]).isEqualTo(1.1);
    }

    @Test
    void convolution2dAndBNWithFullParamsShouldBuildLayers() {
        ComputationGraphConfiguration.GraphBuilder graph = createGraphBuilder();
        FaceNetSmallV2Helper.convolution2dAndBN(graph, "test_block",
                128, 256, new int[]{1, 1}, new int[]{1, 1},
                256, 128, new int[]{3, 3}, new int[]{2, 2},
                new int[]{1, 1, 1, 1}, "input");
        assertThat(FaceNetSmallV2Model.reluIndex).isGreaterThan(1);
    }

    @Test
    void convolution2dAndBNWithNullConv2ShouldBuildSingleConv() {
        ComputationGraphConfiguration.GraphBuilder graph = createGraphBuilder();
        FaceNetSmallV2Helper.convolution2dAndBN(graph, "test_block",
                128, 256, new int[]{1, 1}, new int[]{1, 1},
                null, null, null, null,
                new int[]{1, 1, 1, 1}, "input");
        assertThat(FaceNetSmallV2Model.reluIndex).isGreaterThan(1);
    }

    @Test
    void convolution2dAndBNWithNullPaddingShouldSkipPadding() {
        ComputationGraphConfiguration.GraphBuilder graph = createGraphBuilder();
        FaceNetSmallV2Helper.convolution2dAndBN(graph, "test_block",
                128, 256, new int[]{1, 1}, new int[]{1, 1},
                256, 128, new int[]{3, 3}, new int[]{2, 2},
                null, "input");
        assertThat(FaceNetSmallV2Model.reluIndex).isGreaterThan(1);
    }

    @Test
    void convolution2dAndBNWithNullConv2AndNullPadding() {
        ComputationGraphConfiguration.GraphBuilder graph = createGraphBuilder();
        FaceNetSmallV2Helper.convolution2dAndBN(graph, "test_block",
                128, 256, new int[]{1, 1}, new int[]{1, 1},
                null, null, null, null,
                null, "input");
        assertThat(FaceNetSmallV2Model.reluIndex).isGreaterThan(1);
    }

    private ComputationGraphConfiguration.GraphBuilder createGraphBuilder() {
        return new NeuralNetConfiguration.Builder()
                .seed(1234)
                .activation(Activation.IDENTITY)
                .weightInit(WeightInit.RELU)
                .updater(new Adam(0.1, 0.9, 0.999, 0.01))
                .l2(5e-5)
                .miniBatch(true)
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(96, 96, 3));
    }
}
