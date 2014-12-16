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
package hivemall.mix.client;

import hivemall.mix.MixMessage;
import hivemall.mix.MixedModel;
import io.netty.channel.ChannelHandler.Sharable;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;

@Sharable
public final class MixClientHandler extends SimpleChannelInboundHandler<MixMessage> {

    private final MixedModel model;

    public MixClientHandler(MixedModel model) {
        super();
        if(model == null) {
            throw new IllegalArgumentException("model is null");
        }
        this.model = model;
    }

    @Override
    protected void channelRead0(ChannelHandlerContext ctx, MixMessage msg) throws Exception {
        Object feature = msg.getFeature();
        float weight = msg.getWeight();
        short clock = msg.getClock();
        float covar = msg.getCovariance();
        model.set(feature, weight, covar, clock);
    }

}
