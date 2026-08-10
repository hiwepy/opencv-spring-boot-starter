package org.bytedeco.opencv.spring.boot.nd4j;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Tests for {@link FaceRecognition}.
 * @author <a href="https://github.com/loong10k">Loong Wan</a>
 */
class FaceRecognitionTest {

    @Test
    void shouldCreateInstance() {
        FaceRecognition faceRecognition = new FaceRecognition();
        assertThat(faceRecognition).isNotNull();
    }
}
