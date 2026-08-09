package org.bytedeco.opencv.spring.boot.nd4j;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Tests for {@link Nd4jImageLoaderProperties}.
 * @author [@Loong Wan](https://github.com/loong10k)
 */
class Nd4jImageLoaderPropertiesTest {

    @Test
    void shouldHaveCorrectPrefix() {
        assertThat(Nd4jImageLoaderProperties.PREFIX).isEqualTo("opencv.nd4j.loader.default");
    }

    @Test
    void shouldHaveDefaultValues() {
        Nd4jImageLoaderProperties props = new Nd4jImageLoaderProperties();
        assertThat(props.getHeight()).isEqualTo(-1L);
        assertThat(props.getWidth()).isEqualTo(-1L);
        assertThat(props.getChannels()).isEqualTo(-1L);
        assertThat(props.isCenterCropIfNeeded()).isFalse();
    }

    @Test
    void shouldSetAndGetHeight() {
        Nd4jImageLoaderProperties props = new Nd4jImageLoaderProperties();
        props.setHeight(224L);
        assertThat(props.getHeight()).isEqualTo(224L);
    }

    @Test
    void shouldSetAndGetWidth() {
        Nd4jImageLoaderProperties props = new Nd4jImageLoaderProperties();
        props.setWidth(224L);
        assertThat(props.getWidth()).isEqualTo(224L);
    }

    @Test
    void shouldSetAndGetChannels() {
        Nd4jImageLoaderProperties props = new Nd4jImageLoaderProperties();
        props.setChannels(3L);
        assertThat(props.getChannels()).isEqualTo(3L);
    }

    @Test
    void shouldSetAndGetCenterCropIfNeeded() {
        Nd4jImageLoaderProperties props = new Nd4jImageLoaderProperties();
        props.setCenterCropIfNeeded(true);
        assertThat(props.isCenterCropIfNeeded()).isTrue();
    }
}
