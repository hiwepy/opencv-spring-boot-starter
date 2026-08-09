package org.bytedeco.opencv.spring.boot.nd4j;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Tests for {@link Nd4NativeLoaderProperties}.
 * @author [@Loong Wan](https://github.com/loong10k)
 */
class Nd4NativeLoaderPropertiesTest {

    @Test
    void shouldHaveCorrectPrefix() {
        assertThat(Nd4NativeLoaderProperties.PREFIX).isEqualTo("opencv.nd4j.loader.native");
    }

    @Test
    void shouldHaveDefaultValues() {
        Nd4NativeLoaderProperties props = new Nd4NativeLoaderProperties();
        assertThat(props.getHeight()).isEqualTo(99L);
        assertThat(props.getWidth()).isEqualTo(99L);
        assertThat(props.getChannels()).isEqualTo(3L);
        assertThat(props.isCenterCropIfNeeded()).isFalse();
    }

    @Test
    void shouldSetAndGetHeight() {
        Nd4NativeLoaderProperties props = new Nd4NativeLoaderProperties();
        props.setHeight(224L);
        assertThat(props.getHeight()).isEqualTo(224L);
    }

    @Test
    void shouldSetAndGetWidth() {
        Nd4NativeLoaderProperties props = new Nd4NativeLoaderProperties();
        props.setWidth(224L);
        assertThat(props.getWidth()).isEqualTo(224L);
    }

    @Test
    void shouldSetAndGetChannels() {
        Nd4NativeLoaderProperties props = new Nd4NativeLoaderProperties();
        props.setChannels(1L);
        assertThat(props.getChannels()).isEqualTo(1L);
    }

    @Test
    void shouldSetAndGetCenterCropIfNeeded() {
        Nd4NativeLoaderProperties props = new Nd4NativeLoaderProperties();
        props.setCenterCropIfNeeded(true);
        assertThat(props.isCenterCropIfNeeded()).isTrue();
    }
}
