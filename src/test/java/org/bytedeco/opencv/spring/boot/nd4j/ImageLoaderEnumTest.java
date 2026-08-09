package org.bytedeco.opencv.spring.boot.nd4j;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Tests for {@link ImageLoader} enum.
 * @author [@Loong Wan](https://github.com/loong10k)
 */
class ImageLoaderEnumTest {

    @Test
    void shouldHaveAllExpectedValues() {
        ImageLoader[] values = ImageLoader.values();
        assertThat(values).hasSize(4);
        assertThat(values).containsExactly(
                ImageLoader.CIFAR,
                ImageLoader.DEFAULT,
                ImageLoader.LFW,
                ImageLoader.NATIVE
        );
    }

    @Test
    void shouldBeAccessibleByValueOf() {
        assertThat(ImageLoader.valueOf("CIFAR")).isEqualTo(ImageLoader.CIFAR);
        assertThat(ImageLoader.valueOf("DEFAULT")).isEqualTo(ImageLoader.DEFAULT);
        assertThat(ImageLoader.valueOf("LFW")).isEqualTo(ImageLoader.LFW);
        assertThat(ImageLoader.valueOf("NATIVE")).isEqualTo(ImageLoader.NATIVE);
    }
}
