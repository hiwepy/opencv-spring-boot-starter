package org.bytedeco.opencv.spring.boot.image;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Tests for {@link Rect}.
 * @author [@Loong Wan](https://github.com/loong10k)
 */
class RectTest {

    @Test
    void shouldCreateWithDefaultConstructor() {
        Rect rect = new Rect();
        assertThat(rect.getLeft()).isEqualTo(0);
        assertThat(rect.getTop()).isEqualTo(0);
        assertThat(rect.getRight()).isEqualTo(0);
        assertThat(rect.getBottom()).isEqualTo(0);
    }

    @Test
    void shouldCreateWithParameterizedConstructor() {
        Rect rect = new Rect(10, 20, 30, 40);
        assertThat(rect.getLeft()).isEqualTo(10);
        assertThat(rect.getTop()).isEqualTo(20);
        assertThat(rect.getRight()).isEqualTo(30);
        assertThat(rect.getBottom()).isEqualTo(40);
    }

    @Test
    void shouldDeepCopyWithCopyConstructor() {
        Rect original = new Rect(10, 20, 30, 40);
        Rect copy = new Rect(original);
        assertThat(copy.getLeft()).isEqualTo(10);
        assertThat(copy.getTop()).isEqualTo(20);
        assertThat(copy.getRight()).isEqualTo(30);
        assertThat(copy.getBottom()).isEqualTo(40);
    }

    @Test
    void shouldHandleNullInCopyConstructor() {
        Rect copy = new Rect(null);
        assertThat(copy.getLeft()).isEqualTo(0);
        assertThat(copy.getTop()).isEqualTo(0);
        assertThat(copy.getRight()).isEqualTo(0);
        assertThat(copy.getBottom()).isEqualTo(0);
    }

    @Test
    void shouldSetAndGetLeft() {
        Rect rect = new Rect();
        rect.setLeft(5);
        assertThat(rect.getLeft()).isEqualTo(5);
    }

    @Test
    void shouldSetAndGetTop() {
        Rect rect = new Rect();
        rect.setTop(15);
        assertThat(rect.getTop()).isEqualTo(15);
    }

    @Test
    void shouldSetAndGetRight() {
        Rect rect = new Rect();
        rect.setRight(25);
        assertThat(rect.getRight()).isEqualTo(25);
    }

    @Test
    void shouldSetAndGetBottom() {
        Rect rect = new Rect();
        rect.setBottom(35);
        assertThat(rect.getBottom()).isEqualTo(35);
    }

    @Test
    void shouldFormatToString() {
        Rect rect = new Rect(1, 2, 3, 4);
        String str = rect.toString();
        assertThat(str).contains("1");
        assertThat(str).contains("2");
        assertThat(str).contains("3");
        assertThat(str).contains("4");
    }
}
