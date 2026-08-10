package org.bytedeco.opencv.spring.boot;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Tests for {@link OpenCVFaceRecognitionProperties}.
 * @author <a href="https://github.com/loong10k">Loong Wan</a>
 */
class OpenCVFaceRecognitionPropertiesTest {

    @Test
    void shouldHaveCorrectPrefix() {
        assertThat(OpenCVFaceRecognitionProperties.PREFIX).isEqualTo("opencv.face");
    }

    @Test
    void shouldHaveDefaultValues() {
        OpenCVFaceRecognitionProperties props = new OpenCVFaceRecognitionProperties();
        assertThat(props.isEnabled()).isFalse();
        assertThat(props.getTemp()).isNotNull();
        assertThat(props.getTemp()).isNotEmpty();
    }

    @Test
    void shouldSetAndGetEnabled() {
        OpenCVFaceRecognitionProperties props = new OpenCVFaceRecognitionProperties();
        props.setEnabled(true);
        assertThat(props.isEnabled()).isTrue();
    }

    @Test
    void shouldSetAndGetTemp() {
        OpenCVFaceRecognitionProperties props = new OpenCVFaceRecognitionProperties();
        props.setTemp("/tmp/opencv");
        assertThat(props.getTemp()).isEqualTo("/tmp/opencv");
    }
}
