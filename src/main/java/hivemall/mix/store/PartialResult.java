/*
 * Hivemall: Hive scalable Machine Learning Library
 *
 * Copyright (C) 2013-2014
 *   National Institute of Advanced Industrial Science and Technology (AIST)
 *   Registration Number: H25PRO-1520
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */
package hivemall.mix.store;

import hivemall.utils.lock.Lock;
import hivemall.utils.lock.TTASLock;

import javax.annotation.Nonnegative;
import javax.annotation.concurrent.GuardedBy;

public abstract class PartialResult {
    protected static final int ACCUMULATE_THRESHOLD = 100;

    private final Lock lock;

    @GuardedBy("lock()")
    protected float minCovariance;
    @GuardedBy("lock()")
    protected short totalClock;
    @GuardedBy("lock()")
    protected float weight;

    public PartialResult() {
        this.minCovariance = 1.0f;
        this.totalClock = 0;
        this.weight = 0.f;
        this.lock = new TTASLock();
    }

    public final void lock() {
        lock.lock();
    }

    public final void unlock() {
        lock.unlock();
    }

    public abstract void add(float localWeight, float covar, short clock, @Nonnegative int deltaUpdates);

    public final float getMinCovariance() {
        return minCovariance;
    }

    protected final void setMinCovariance(float covar) {
        this.minCovariance = Math.min(minCovariance, covar);
    }

    public final short getClock() {
        return totalClock;
    }

    protected final void incrClock(short clock) {
        totalClock += clock;
    }

    public final int diffClock(short clock) {
        short diff = (short) (totalClock - clock);
        return diff < 0 ? -diff : diff;
    }

    public final float getWeight() {
        accumulate();
        return weight;
    }

    protected abstract void accumulate();

    /**
     * Weighting by exponential moving average.
     * 
     * http://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
     */
    protected final void updateWeight(final float newValue) {
        final float oldWeight = weight;
        if(oldWeight == 0.f) {
            this.weight = newValue;
        } else {
            this.weight = 0.3f * newValue + 0.7f * oldWeight;
        }
    }

}
