
define([
    'decorators/ModelDecorator/Core/Port'
], function (Port) {

    'use strict';

    var BlkPort,
        PORT_CONNECTOR_LEN = 20,
        PORT_DOM_HEIGHT = 15,
        PORT_TITLE_WRAPPER_WITH_ICON_CLASS = 'w-icon',
        PORT_DOT_WIDTH = 3,
        PORT_ICON_WIDTH = 5;   //_Port.scss: $svg-icon-width: 5px;

    BlkPort = function (id, options) {
        var opts = {
            title: options.title,
            decorator: options.decorator,
            svg: options.svg
        };
        var opts1 = _.extend({}, options);
        Port.apply(this, [id, options]);
        this.logger.debug('BlkDecorator::BlkPort created');

    };

    _.extend(BlkPort.prototype, Port.prototype);




    BlkPort.prototype.getConnectorArea = function () {
        var allPorts = this.$el.parent().children(),
            len = allPorts.length,
            i;

        for (i = 0; i < len; i += 1) {
            if (allPorts[i] === this.$el[0]) {
                break;
            }
        }

        this.connectionArea.x1 = this.orientation === 'W' ? 0 : this.decorator.hostDesignerItem.getWidth();
        //fix the x coordinate for the dot/svg icon's width
        if (this.icon) {
            this.connectionArea.x1 += (this.orientation === 'W' ? -1 : 1) * (PORT_ICON_WIDTH - 1);
        } else {
            this.connectionArea.x1 += (this.orientation === 'W' ? -1 : 1) * (PORT_DOT_WIDTH - 1);
        }
        this.connectionArea.x2 = this.connectionArea.x1;
        this.connectionArea.y1 = i * PORT_DOM_HEIGHT + 30;
        this.connectionArea.y2 = this.connectionArea.y1;
        this.connectionArea.angle1 = this.orientation === 'W' ? 180 : 0;
        this.connectionArea.angle2 = this.orientation === 'W' ? 180 : 0;

        return this.connectionArea;
    };

    return BlkPort;
});