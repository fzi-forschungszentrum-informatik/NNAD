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

#include "utils.hh"

#include "dataset.hh"

std::shared_ptr<DatasetEntry> SequentialDataset::getNext()
{
    std::scoped_lock<std::mutex> lockGuard(m_idxMutex);

    if (m_idx >= num()) {
        return {nullptr};
    }

    return get(m_idx++);
}


ParallelDataset::ParallelDataset(std::shared_ptr<Dataset> inputDataset, int numWorkers, std::size_t queueSize)
    : m_inputDataset(inputDataset), m_queueSize(queueSize), m_run(true), m_numWorkers(numWorkers)
{
    for (int i = 0; i < numWorkers; ++i) {
        m_threads.emplace_back(std::bind(&ParallelDataset::worker, this));
    }
}

ParallelDataset::~ParallelDataset()
{
    m_run = false;
    for (auto &thread : m_threads)
    {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

std::shared_ptr<DatasetEntry> ParallelDataset::getNext()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    while (m_queue.size() == 0) {
        m_popCondVar.wait_for(lock, std::chrono::seconds(5));
        {
            std::scoped_lock lock(m_numWorkersMutex);
            if (m_queue.size() == 0 && m_numWorkers == 0) {
                return nullptr;
            }
        }
    }
    std::shared_ptr<DatasetEntry> data;
    {
        std::scoped_lock<std::mutex> lockGuard(m_accessMutex);
        data = m_queue.front();
        m_queue.pop();
    }
    m_pushCondVar.notify_one();
    return data;
}

void ParallelDataset::worker()
{
    while (m_run) {
        auto data = m_inputDataset->getNext();
        if (data) {
            std::unique_lock<std::mutex> lock(m_mutex);
            {
                std::scoped_lock<std::mutex> lockGuard(m_accessMutex);
                m_queue.push(data);
            }
            m_popCondVar.notify_one();
            while (m_queue.size() >= m_queueSize) {
                m_pushCondVar.wait_for(lock, std::chrono::seconds(5));
                if (!m_run) {
                    break;
                }
            }
        } else {
            break;
        }
    }
    {
        std::scoped_lock lock(m_numWorkersMutex);
        --m_numWorkers;
    }
}


RandomDataset::RandomDataset(std::shared_ptr<SequentialDataset> inputDataset)
    : m_inputDataset(inputDataset)
{
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    m_generator = std::mt19937(seed);
    m_indexDistribution = std::uniform_int_distribution<std::size_t>(0, m_inputDataset->num() - 1);
}

std::shared_ptr<DatasetEntry> RandomDataset::getNext()
{
    int64_t idx;
    {
        std::scoped_lock<std::mutex> lockGuard(m_generatorMutex);
        idx = m_indexDistribution(m_generator);
    }
    return m_inputDataset->get(idx);
}


InterleaveDataset::InterleaveDataset(std::vector<std::shared_ptr<Dataset>> inputDatasets)
    : m_inputDatasets(inputDatasets), m_activeInputDatasets(inputDatasets)
{
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    m_generator = std::mt19937(seed);
    CHECK(m_inputDatasets.size() > 0, "You must add at least one dataset");
    m_indexDistribution = std::uniform_int_distribution<std::size_t>(0, m_inputDatasets.size() - 1);
}

std::shared_ptr<DatasetEntry> InterleaveDataset::getNext()
{
    while (true) {
        std::size_t listSize;
        {
            std::scoped_lock<std::mutex> lockGuard(m_datasetMutex);
            listSize = m_activeInputDatasets.size();
            if (listSize == 0) {
                return nullptr;
            }
        }
        int64_t idx;
        {
            std::scoped_lock<std::mutex> lockGuard(m_generatorMutex);
            idx = m_indexDistribution(m_generator);
        }
        auto data = m_activeInputDatasets[idx]->getNext();
        if (data) {
            return data;
        } else {
            std::scoped_lock<std::mutex> lockGuard(m_datasetMutex);
            if (listSize == m_activeInputDatasets.size()) {
                m_activeInputDatasets.erase(m_activeInputDatasets.begin() + idx);
            }
        }
    }
}


MapDataset::MapDataset(std::shared_ptr<Dataset> inputDataset, std::function<void(std::shared_ptr<DatasetEntry>)> mapFn)
    : m_inputDataset(inputDataset), m_mapFn(mapFn)
{
}

std::shared_ptr<DatasetEntry> MapDataset::getNext()
{
    auto data = m_inputDataset->getNext();
    m_mapFn(data);
    return data;
}
