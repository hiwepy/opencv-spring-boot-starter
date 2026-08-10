package org.bytedeco.opencv.spring.boot.nd4j;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Tests for {@link Nd4jLFWLoaderProperties}.
 * @author <a href="https://github.com/loong10k">Loong Wan</a>
 */
class Nd4jLFWLoaderPropertiesTest {

    @Test
    void shouldHaveCorrectPrefix() {
        assertThat(Nd4jLFWLoaderProperties.PREFIX).isEqualTo("opencv.nd4j.loader.lfw");
    }

    @Test
    void shouldHaveDefaultValues() {
        Nd4jLFWLoaderProperties props = new Nd4jLFWLoaderProperties();
        assertThat(props.getHeight()).isEqualTo(org.datavec.image.loader.LFWLoader.HEIGHT);
        assertThat(props.getWidth()).isEqualTo(org.datavec.image.loader.LFWLoader.WIDTH);
        assertThat(props.getChannels()).isEqualTo(org.datavec.image.loader.LFWLoader.CHANNELS);
        assertThat(props.isUseSubset()).isFalse();
    }

    @Test
    void shouldSetAndGetHeight() {
        Nd4jLFWLoaderProperties props = new Nd4jLFWLoaderProperties();
        props.setHeight(128L);
        assertThat(props.getHeight()).isEqualTo(128L);
    }

    @Test
    void shouldSetAndGetWidth() {
        Nd4jLFWLoaderProperties props = new Nd4jLFWLoaderProperties();
        props.setWidth(128L);
        assertThat(props.getWidth()).isEqualTo(128L);
    }

    @Test
    void shouldSetAndGetChannels() {
        Nd4jLFWLoaderProperties props = new Nd4jLFWLoaderProperties();
        props.setChannels(1L);
        assertThat(props.getChannels()).isEqualTo(1L);
    }

    @Test
    void shouldSetAndGetUseSubset() {
        Nd4jLFWLoaderProperties props = new Nd4jLFWLoaderProperties();
        props.setUseSubset(true);
        assertThat(props.isUseSubset()).isTrue();
    }
}
