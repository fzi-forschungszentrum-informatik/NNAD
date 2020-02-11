/**************************************************************************
 * NNAD (Neural Networks for Automated Driving) training scripts          *
 * Copyright (C) 2019 FZI Research Center for Information Technology      *
 *                                                                        *
 * This program is free software: you can redistribute it and/or modify   *
 * it under the terms of the GNU General Public License as published by   *
 * the Free Software Foundation, either version 3 of the License, or      *
 * (at your option) any later version.                                    *
 *                                                                        *
 * This program is distributed in the hope that it will be useful,        *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of         *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          *
 * GNU General Public License for more details.                           *
 *                                                                        *
 * You should have received a copy of the GNU General Public License      *
 * along with this program.  If not, see <https://www.gnu.org/licenses/>. *
 **************************************************************************/

#pragma once

#include <string>
#include <map>
#include <memory>
#include <variant>
#include <opencv2/core/mat.hpp>

struct BoundingBox {
    BoundingBox()
    {
    }
    int64_t id;
    int32_t cls;
    int32_t x1;
    int32_t y1;
    int32_t x2;
    int32_t y2;
};

struct BoundingBoxDetection {
    BoundingBox box;
    float score;
    std::vector<float> embedding;
};

struct AnchorBox {
    /* Box center */
    int32_t xc;
    int32_t yc;

    /* Top-left corner */
    int32_t x1;
    int32_t y1;

    /* Bottom-right corner */
    int32_t x2;
    int32_t y2;
};

enum Objectness {
    DONT_CARE = -1,
    NO_OBJECT = 0,
    OBJECT = 1
};

struct TargetBox {
    TargetBox() : id(0), dxc(0.0), dyc(0.0), dw(0.0), dh(0.0), cls(0), objectness(Objectness::NO_OBJECT)
    {
    }

    /* Box id */
    int64_t id;

    /* See Appendix C of "Rich feature hierarchies for accurate object detection and semantic segmentation" */
    double dxc;
    double dyc;
    double dw;
    double dh;

    /* Class */
    int32_t cls;

    /* Objectness */
    int32_t objectness;
};

struct TargetBoxDetection {
    TargetBoxDetection() : dxc(0.0), dyc(0.0), dw(0.0), dh(0.0), cls(0), objectnessScore(0.0)
    {
    }

    /* See Appendix C of "Rich feature hierarchies for accurate object detection and semantic segmentation" */
    double dxc;
    double dyc;
    double dw;
    double dh;

    /* Class */
    int32_t cls;

    std::vector<float> embedding;

    /* Objectness */
    double objectnessScore;
};

struct BoundingBoxList {
    BoundingBoxList() : width(0), height(0), valid(false)
    {
    }

    int32_t width;
    int32_t height;
    bool valid;
    std::vector<BoundingBox> boxes;
    std::vector<std::vector<TargetBox>> targets;
};

struct DatasetEntry {
    struct {
        cv::Mat left;
    } input;

    struct {
        cv::Mat pixelwiseLabels;
        cv::Mat bbDontCareAreas;
        BoundingBoxList bbList;
        int cls; // For pretraining
    } gt;

    struct {
        std::string key;
        int32_t originalWidth;
        int32_t originalHeight;
        bool canFlip;
        double horizontalFov;
    } metadata;
};
