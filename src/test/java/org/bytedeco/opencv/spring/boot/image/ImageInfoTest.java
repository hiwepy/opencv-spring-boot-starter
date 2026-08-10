package org.bytedeco.opencv.spring.boot.image;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Tests for {@link ImageInfo}.
 * @author <a href="https://github.com/loong10k">Loong Wan</a>
 */
class ImageInfoTest {

    @Test
    void shouldCreateEmptyImageInfo() {
        ImageInfo info = new ImageInfo();
        assertThat(info.getImageData()).isNull();
        assertThat(info.getWidth()).isNull();
        assertThat(info.getHeight()).isNull();
        assertThat(info.getImageFormat()).isNull();
    }

    @Test
    void shouldSetAndGetImageData() {
        ImageInfo info = new ImageInfo();
        byte[] data = new byte[]{1, 2, 3, 4};
        info.setImageData(data);
        assertThat(info.getImageData()).isEqualTo(data);
    }

    @Test
    void shouldSetAndGetWidth() {
        ImageInfo info = new ImageInfo();
        info.setWidth(640);
        assertThat(info.getWidth()).isEqualTo(640);
    }

    @Test
    void shouldSetAndGetHeight() {
        ImageInfo info = new ImageInfo();
        info.setHeight(480);
        assertThat(info.getHeight()).isEqualTo(480);
    }

    @Test
    void shouldSetAndGetImageFormat() {
        ImageInfo info = new ImageInfo();
        info.setImageFormat(ImageFormat.CP_PAF_BGR24);
        assertThat(info.getImageFormat()).isEqualTo(ImageFormat.CP_PAF_BGR24);
    }

    @Test
    void shouldImplementEqualsAndHashCode() {
        ImageInfo info1 = new ImageInfo();
        info1.setWidth(100);
        info1.setHeight(200);
        info1.setImageFormat(ImageFormat.CP_PAF_GRAY);

        ImageInfo info2 = new ImageInfo();
        info2.setWidth(100);
        info2.setHeight(200);
        info2.setImageFormat(ImageFormat.CP_PAF_GRAY);

        assertThat(info1).isEqualTo(info2);
        assertThat(info1.hashCode()).isEqualTo(info2.hashCode());
    }

    @Test
    void shouldImplementToString() {
        ImageInfo info = new ImageInfo();
        info.setWidth(320);
        assertThat(info.toString()).contains("320");
    }
}
