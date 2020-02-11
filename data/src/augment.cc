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

#include <opencv2/imgproc/imgproc.hpp>
#include <random>
#include <chrono>

#include "utils.hh"

#include "augment.hh"

PointwiseDistort::PointwiseDistort()
{
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    m_generator = std::mt19937(seed);
}

void PointwiseDistort::apply(std::shared_ptr<DatasetEntry> data)
{
    double gamma, brightness, contrast, saturation, hue, deltaA, deltaB, blur, noiseSigma;
    {
        std::lock_guard<std::mutex> lockGuard(m_generatorMutex);

        gamma = m_gammaDistribution(m_generator);
        brightness = m_brightnessDistribution(m_generator);
        contrast = m_contrastDistribution(m_generator);
        saturation = m_saturationDistribution(m_generator);
        hue = m_hueDistribution(m_generator);
        deltaA = m_deltaColorDistribution(m_generator);
        deltaB = m_deltaColorDistribution(m_generator);
        blur = m_blurDistribution(m_generator);
        noiseSigma = m_noiseSigmaDistribution(m_generator);
    }

    distortImage(data->input.left, gamma, brightness, contrast, saturation, hue, deltaA, deltaB, blur, noiseSigma);
}

void PointwiseDistort::distortImage(cv::Mat &img, double gamma, double brightness, double contrast,
    double saturation, double hue, double deltaA, double deltaB, double blur, double noiseSigma)
{
    /* Apply random gamma */
    cv::pow(img, gamma, img);

    /* Apply random contrast */
    img *= contrast;

    /* Apply random brightness */
    img += cv::Scalar(brightness, brightness, brightness);

    /* Random hue and saturation */
    img = cv::min(1.0, cv::max(0.0, img));
    cv::cvtColor(img, img, CV_BGR2HSV);
    img += cv::Scalar(hue, saturation, 0.0);
    cv::cvtColor(img, img, CV_HSV2BGR);

    /* Random white balance */
    cv::cvtColor(img, img, CV_BGR2Lab);
    img += cv::Scalar(0.0, deltaA, deltaB);
    cv::cvtColor(img, img, CV_Lab2BGR);

    /* Gaussian blur */
    cv::GaussianBlur(img, img, cv::Size(0, 0), blur);

    /* Add Gaussian noise to input images */
    cv::Mat noise(img.size(), img.type());
    cv::randn(noise, cv::Scalar(0.0, 0.0, 0.0), cv::Scalar(noiseSigma, noiseSigma, noiseSigma));
    img += noise;

    img = cv::min(1.0, cv::max(0.0, img));
}

CropResize::CropResize(cv::Size newSize, double desiredHorizontalFov, bool allowFlip, double deltaCrop)
    : m_newSize(newSize), m_desiredHorizontalFov(desiredHorizontalFov), m_allowFlip(allowFlip), m_deltaCrop(deltaCrop)
{
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    m_generator = std::mt19937(seed);
}

void CropResize::apply(std::shared_ptr<DatasetEntry> data)
{
    bool doFlip = false;
    if (data->metadata.canFlip && m_allowFlip) {
        std::lock_guard<std::mutex> lockGuard(m_generatorMutex);
        doFlip = m_flipDistribution(m_generator) == 1;
    }

    /* Select a ROI for the crop */
    auto &firstImg = data->input.left;
    int x = 0;
    int y = 0;
    int newWidth = firstImg.cols;
    int newHeight = firstImg.rows;

    if (m_desiredHorizontalFov > 0.0) {
        auto currentHorizontalFov = data->metadata.horizontalFov;
        /* Here we assume a spherical undistort */
        double meanDesiredWidth = firstImg.cols * m_desiredHorizontalFov / currentHorizontalFov;

        CHECK(meanDesiredWidth < firstImg.cols, "The desired mean width must not be larger than the input size");
        double lowerWidth = meanDesiredWidth * (1.0 - 0.5 * m_deltaCrop);
        double upperWidth = meanDesiredWidth * (1.0 + 0.5 * m_deltaCrop);
        CHECK(lowerWidth > 50, "The lower crop must be reasonably large (> 50) but is "
              + std::to_string(lowerWidth) + " pixels large");
        CHECK(lowerWidth * m_newSize.height / m_newSize.width > 50,
              "The lower crop must be reasonably large (> 50) but is "
              + std::to_string(lowerWidth * m_newSize.height / m_newSize.width) + " pixels large");
        CHECK(upperWidth < firstImg.cols, "The upper crop must not be larger than the input size");
        CHECK(upperWidth * m_newSize.height / m_newSize.width < firstImg.rows,
              "The upper crop must not be larger than the input size");
        std::uniform_int_distribution<int> widthDistribution(lowerWidth, upperWidth);

        {
            std::lock_guard<std::mutex> lockGuard(m_generatorMutex);

            newWidth = widthDistribution(m_generator);
            newHeight = newWidth * static_cast<double>(m_newSize.height) / static_cast<double>(m_newSize.width);
            std::uniform_int_distribution<int> xDistribution(0, firstImg.cols - newWidth);
            std::uniform_int_distribution<int> yDistribution(0, firstImg.rows - newHeight);
            x = xDistribution(m_generator);
            y = yDistribution(m_generator);
        }
    }
    cv::Rect roi(x, y, newWidth, newHeight);

    handleInputImage(data->input.left, roi, doFlip);
    handleGtImage(data->gt.pixelwiseLabels, roi, doFlip);
    handleGtImage(data->gt.bbDontCareAreas, roi, doFlip);
    handleBoundingBoxList(data->gt.bbList, roi, doFlip);
}

void CropResize::handleInputImage(cv::Mat &img, cv::Rect roi, bool flip) const
{
    if (!img.data) {
        return;
    }

    /* Crop the roi */
    img = img(roi);

    if (flip) {
        cv::flip(img, img, 1);
    }

    /* Resize the image to the new size */
    cv::resize(img, img, m_newSize);
}

void CropResize::handleGtImage(cv::Mat &img, cv::Rect roi, bool flip) const
{
    if (!img.data) {
        return;
    }

    /* Crop the roi */
    img = img(roi);

    if (flip) {
        cv::flip(img, img, 1);
    }

    /* Resize the image to the new size */
    cv::resize(img, img, m_newSize, 0, 0, cv::INTER_NEAREST);
}

void CropResize::handleBoundingBoxList(BoundingBoxList &boxList, cv::Rect roi, bool flip) const
{
    if (!boxList.valid) {
        return;
    }
    double scaleX = static_cast<double>(m_newSize.width) / static_cast<double>(roi.width);
    double scaleY = static_cast<double>(m_newSize.height) / static_cast<double>(roi.height);
    boxList.width = m_newSize.width;
    boxList.height = m_newSize.height;

    auto handleBox = [&] (BoundingBox &box) {
        /* Handle roi crop */
        box.x1 -= roi.x;
        box.x2 -= roi.x;
        box.y1 -= roi.y;
        box.y2 -= roi.y;

        /* Handle flip */
        if (flip) {
            auto x1 = roi.width - 1 - box.x1;
            auto x2 = roi.width - 1 - box.x2;
            /* x1 and x2 have to be swapped now */
            box.x1 = x2;
            box.x2 = x1;
        }
        /* Clip boxes to cropped image size */
        box.x1 = std::clamp(box.x1, 0, roi.width - 1);
        box.x2 = std::clamp(box.x2, 0, roi.width - 1);
        box.y1 = std::clamp(box.y1, 0, roi.height - 1);
        box.y2 = std::clamp(box.y2, 0, roi.height - 1);
        /* Resize to new image size */
        box.x1 *= scaleX;
        box.x2 *= scaleX;
        box.y1 *= scaleY;
        box.y2 *= scaleY;
    };

    for (auto &box : boxList.boxes) {
        handleBox(box);
    }


    /* Delete invalid boxes. */
    boxList.boxes.erase(std::remove_if(boxList.boxes.begin(), boxList.boxes.end(), 
        [](BoundingBox &box) { return box.x1 >= box.x2 || box.y1 >= box.y2; }), boxList.boxes.end());
}
