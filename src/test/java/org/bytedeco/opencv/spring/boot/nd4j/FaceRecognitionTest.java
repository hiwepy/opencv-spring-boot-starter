package org.bytedeco.opencv.spring.boot.nd4j;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Tests for {@link FaceRecognition}.
 * @author [@Loong Wan](https://github.com/loong10k)
 */
class FaceRecognitionTest {

    @Test
    void shouldCreateInstance() {
        FaceRecognition faceRecognition = new FaceRecognition();
        assertThat(faceRecognition).isNotNull();
    }
}
