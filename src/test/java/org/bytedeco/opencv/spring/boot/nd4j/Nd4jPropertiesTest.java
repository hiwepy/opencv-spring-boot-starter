package org.bytedeco.opencv.spring.boot.nd4j;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;

/**
 * Tests for {@link Nd4jProperties}.
 * @author <a href="https://github.com/loong10k">Loong Wan</a>
 */
class Nd4jPropertiesTest {

    @Test
    void shouldHaveCorrectPrefix() {
        assertThat(Nd4jProperties.PREFIX).isEqualTo("opencv.nd4j");
    }

    @Test
    void shouldHaveDefaultValues() {
        Nd4jProperties props = new Nd4jProperties();
        assertThat(props.isMiniBatch()).isTrue();
        assertThat(props.getMaxNumLineSearchIterations()).isEqualTo(0);
        assertThat(props.getSeed()).isEqualTo(0L);
        assertThat(props.getOptimizationAlgo()).isNull();
        assertThat(props.getVariables()).isEmpty();
        assertThat(props.isMinimize()).isTrue();
        assertThat(props.getDataType()).isEqualTo(DataType.FLOAT);
        assertThat(props.getIterationCount()).isEqualTo(0);
        assertThat(props.getEpochCount()).isEqualTo(0);
        assertThat(props.getType()).isEqualTo(ImageLoader.NATIVE);
    }

    @Test
    void shouldSetAndGetMiniBatch() {
        Nd4jProperties props = new Nd4jProperties();
        props.setMiniBatch(false);
        assertThat(props.isMiniBatch()).isFalse();
    }

    @Test
    void shouldSetAndGetSeed() {
        Nd4jProperties props = new Nd4jProperties();
        props.setSeed(42L);
        assertThat(props.getSeed()).isEqualTo(42L);
    }

    @Test
    void shouldSetAndGetDataType() {
        Nd4jProperties props = new Nd4jProperties();
        props.setDataType(DataType.DOUBLE);
        assertThat(props.getDataType()).isEqualTo(DataType.DOUBLE);
    }

    @Test
    void shouldSetAndGetType() {
        Nd4jProperties props = new Nd4jProperties();
        props.setType(ImageLoader.CIFAR);
        assertThat(props.getType()).isEqualTo(ImageLoader.CIFAR);
    }

    @Test
    void shouldSetAndGetIterationCount() {
        Nd4jProperties props = new Nd4jProperties();
        props.setIterationCount(5);
        assertThat(props.getIterationCount()).isEqualTo(5);
    }

    @Test
    void shouldSetAndGetEpochCount() {
        Nd4jProperties props = new Nd4jProperties();
        props.setEpochCount(10);
        assertThat(props.getEpochCount()).isEqualTo(10);
    }
}
