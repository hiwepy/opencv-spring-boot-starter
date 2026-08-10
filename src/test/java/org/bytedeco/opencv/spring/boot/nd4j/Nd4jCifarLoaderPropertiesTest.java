package org.bytedeco.opencv.spring.boot.nd4j;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Tests for {@link Nd4jCifarLoaderProperties}.
 * @author <a href="https://github.com/loong10k">Loong Wan</a>
 */
class Nd4jCifarLoaderPropertiesTest {

    @Test
    void shouldHaveCorrectPrefix() {
        assertThat(Nd4jCifarLoaderProperties.PREFIX).isEqualTo("opencv.nd4j.loader.cifar");
    }

    @Test
    void shouldHaveDefaultValues() {
        Nd4jCifarLoaderProperties props = new Nd4jCifarLoaderProperties();
        assertThat(props.isTrain()).isFalse();
        assertThat(props.getFullPath()).isNull();
    }

    @Test
    void shouldSetAndGetTrain() {
        Nd4jCifarLoaderProperties props = new Nd4jCifarLoaderProperties();
        props.setTrain(true);
        assertThat(props.isTrain()).isTrue();
    }

    @Test
    void shouldSetAndGetFullPath() {
        Nd4jCifarLoaderProperties props = new Nd4jCifarLoaderProperties();
        props.setFullPath("/tmp/cifar");
        assertThat(props.getFullPath()).isEqualTo("/tmp/cifar");
    }
}
