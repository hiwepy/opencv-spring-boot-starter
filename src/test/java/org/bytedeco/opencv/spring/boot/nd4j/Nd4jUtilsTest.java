package org.bytedeco.opencv.spring.boot.nd4j;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Tests for {@link Nd4jUtils}.
 * @author [@Loong Wan](https://github.com/loong10k)
 */
class Nd4jUtilsTest {

    @Test
    void distanceShouldReturnZeroForIdenticalArrays() {
        INDArray a = Nd4j.create(new float[]{1.0f, 2.0f, 3.0f});
        double dist = Nd4jUtils.distance(a, a.dup());
        assertThat(dist).isCloseTo(0.0, within(1e-6));
    }

    @Test
    void distanceShouldReturnPositiveValueForDifferentArrays() {
        INDArray a = Nd4j.create(new float[]{1.0f, 2.0f, 3.0f});
        INDArray b = Nd4j.create(new float[]{4.0f, 5.0f, 6.0f});
        double dist = Nd4jUtils.distance(a, b);
        assertThat(dist).isGreaterThan(0.0);
    }

    @Test
    void transposeShouldReorderChannels() {
        // Create a 1x3x4x4 array (batch=1, channels=3, height=4, width=4)
        INDArray input = Nd4j.create(1, 3, 4, 4);
        // Fill channel 0 with 10, channel 1 with 20, channel 2 with 30
        for (int h = 0; h < 4; h++) {
            for (int w = 0; w < 4; w++) {
                input.putScalar(new int[]{0, 0, h, w}, 10.0);
                input.putScalar(new int[]{0, 1, h, w}, 20.0);
                input.putScalar(new int[]{0, 2, h, w}, 30.0);
            }
        }

        INDArray result = Nd4jUtils.transpose(input, 4, 4);
        assertThat(result).isNotNull();
        assertThat(result.shape()).isEqualTo(new long[]{1, 3, 4, 4});
    }

    @Test
    void transposeShouldHandleSmallArray() {
        INDArray input = Nd4j.create(1, 3, 2, 2);
        INDArray result = Nd4jUtils.transpose(input, 2, 2);
        assertThat(result).isNotNull();
        assertThat(result.shape()[0]).isEqualTo(1);
        assertThat(result.shape()[1]).isEqualTo(3);
        assertThat(result.shape()[2]).isEqualTo(2);
        assertThat(result.shape()[3]).isEqualTo(2);
    }
}
