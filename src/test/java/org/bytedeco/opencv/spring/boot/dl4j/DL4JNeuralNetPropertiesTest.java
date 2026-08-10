package org.bytedeco.opencv.spring.boot.dl4j;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;

/**
 * Tests for {@link DL4JNeuralNetProperties}.
 * @author <a href="https://github.com/loong10k">Loong Wan</a>
 */
class DL4JNeuralNetPropertiesTest {

    @Test
    void shouldHaveCorrectPrefix() {
        assertThat(DL4JNeuralNetProperties.PREFIX).isEqualTo("opencv.dl4j");
    }

    @Test
    void shouldHaveDefaultValues() {
        DL4JNeuralNetProperties props = new DL4JNeuralNetProperties();
        assertThat(props.isMiniBatch()).isTrue();
        assertThat(props.getMaxNumLineSearchIterations()).isEqualTo(0);
        assertThat(props.getSeed()).isEqualTo(0L);
        assertThat(props.getOptimizationAlgo()).isNull();
        assertThat(props.getVariables()).isEmpty();
        assertThat(props.isMinimize()).isTrue();
        assertThat(props.getDataType()).isEqualTo(DataType.FLOAT);
        assertThat(props.getIterationCount()).isEqualTo(0);
        assertThat(props.getEpochCount()).isEqualTo(0);
    }

    @Test
    void shouldSetAndGetMiniBatch() {
        DL4JNeuralNetProperties props = new DL4JNeuralNetProperties();
        props.setMiniBatch(false);
        assertThat(props.isMiniBatch()).isFalse();
    }

    @Test
    void shouldSetAndGetSeed() {
        DL4JNeuralNetProperties props = new DL4JNeuralNetProperties();
        props.setSeed(12345L);
        assertThat(props.getSeed()).isEqualTo(12345L);
    }

    @Test
    void shouldSetAndGetDataType() {
        DL4JNeuralNetProperties props = new DL4JNeuralNetProperties();
        props.setDataType(DataType.DOUBLE);
        assertThat(props.getDataType()).isEqualTo(DataType.DOUBLE);
    }

    @Test
    void shouldSetAndGetIterationCount() {
        DL4JNeuralNetProperties props = new DL4JNeuralNetProperties();
        props.setIterationCount(7);
        assertThat(props.getIterationCount()).isEqualTo(7);
    }

    @Test
    void shouldSetAndGetEpochCount() {
        DL4JNeuralNetProperties props = new DL4JNeuralNetProperties();
        props.setEpochCount(3);
        assertThat(props.getEpochCount()).isEqualTo(3);
    }
}
