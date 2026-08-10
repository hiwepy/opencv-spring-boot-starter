package org.bytedeco.opencv.spring.boot.dl4j;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link FaceNetSmallV2Model}.
 * @author <a href="https://github.com/loong10k">Loong Wan</a>
 */
class FaceNetSmallV2ModelTest {

    @BeforeEach
    void resetIndexes() {
        FaceNetSmallV2Model.reluIndex = 1;
        FaceNetSmallV2Model.paddingIndex = 1;
    }

    @Test
    void shouldCreateInstance() {
        FaceNetSmallV2Model model = new FaceNetSmallV2Model();
        assertThat(model).isNotNull();
    }

    @Test
    void shouldBuildConfiguration() {
        FaceNetSmallV2Model model = new FaceNetSmallV2Model();
        // conf() builds a valid graph configuration when indexes start at 1
        var conf = model.conf();
        assertThat(conf).isNotNull();
    }

    @Test
    void shouldResetIndexesOnConf() {
        FaceNetSmallV2Model model = new FaceNetSmallV2Model();
        // conf() calls resetIndexes() internally and builds the graph
        var conf = model.conf();
        assertThat(conf).isNotNull();
        // After conf(), indexes should have advanced from 1
        assertThat(FaceNetSmallV2Model.reluIndex).isGreaterThan(1);
        assertThat(FaceNetSmallV2Model.paddingIndex).isGreaterThan(1);
    }
}
