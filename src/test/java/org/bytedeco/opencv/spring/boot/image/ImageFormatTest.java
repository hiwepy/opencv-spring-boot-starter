package org.bytedeco.opencv.spring.boot.image;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Tests for {@link ImageFormat}.
 * @author [@Loong Wan](https://github.com/loong10k)
 */
class ImageFormatTest {

    @Test
    void shouldHaveAllExpectedValues() {
        ImageFormat[] values = ImageFormat.values();
        assertThat(values).hasSize(7);
        assertThat(values).containsExactly(
                ImageFormat.CP_PAF_NV21,
                ImageFormat.CP_PAF_NV12,
                ImageFormat.CP_PAF_I420,
                ImageFormat.CP_PAF_YUYV,
                ImageFormat.CP_PAF_BGR24,
                ImageFormat.CP_PAF_GRAY,
                ImageFormat.CP_PAF_DEPTH_U16
        );
    }

    @Test
    void shouldReturnCorrectIntValue() {
        assertThat(ImageFormat.CP_PAF_NV21.getValue()).isEqualTo(2050);
        assertThat(ImageFormat.CP_PAF_NV12.getValue()).isEqualTo(2049);
        assertThat(ImageFormat.CP_PAF_I420.getValue()).isEqualTo(1537);
        assertThat(ImageFormat.CP_PAF_YUYV.getValue()).isEqualTo(1281);
        assertThat(ImageFormat.CP_PAF_BGR24.getValue()).isEqualTo(513);
        assertThat(ImageFormat.CP_PAF_GRAY.getValue()).isEqualTo(1793);
        assertThat(ImageFormat.CP_PAF_DEPTH_U16.getValue()).isEqualTo(3074);
    }

    @Test
    void shouldBeAccessibleByValueOf() {
        assertThat(ImageFormat.valueOf("CP_PAF_NV21")).isEqualTo(ImageFormat.CP_PAF_NV21);
        assertThat(ImageFormat.valueOf("CP_PAF_BGR24")).isEqualTo(ImageFormat.CP_PAF_BGR24);
    }
}
