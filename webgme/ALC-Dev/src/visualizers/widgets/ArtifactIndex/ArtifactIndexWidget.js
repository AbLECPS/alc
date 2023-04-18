/*globals define, $*/
/*jshint browser: true*/

define([
    './ModelItem',
    'text!./Table.html',
    'css!./styles/ArtifactIndexWidget.css'
], function (
    ModelItem,
    TABLE_HTML
) {
    'use strict';

    var ArtifactIndexWidget,
        WIDGET_CLASS = 'artifact-index',
        nop = function(){};

    ArtifactIndexWidget = function (logger, container) {
        this._logger = logger.fork('Widget');

        this.$el = container;

        this.nodes = {};
        this.nodesTime = {};
        this.currentNode = null;
        this._initialize();

        this._logger.debug('ctor finished');
    };

    ArtifactIndexWidget.prototype._initialize = function () {
        // set widget class
        this.$el.addClass(WIDGET_CLASS);
        var dummy = document.createElement('div');
            dummy.id = 'dummyheader';
            dummy.setAttribute("style", "margin-top:25px;");
            this.$el.append(dummy);

        this.$content = $(TABLE_HTML);
        this.$el.append(this.$content);
        this.$list = this.$content.find('.list-content');
    };

    ArtifactIndexWidget.prototype.onWidgetContainerResize = nop;

    // Adding/Removing/Updating items
    ArtifactIndexWidget.prototype.addNode = function (desc) {
        //if (desc && desc.parentId === this.currentNode && (desc.metaName.indexOf('Data')>-1)) {
        if (desc &&  (desc.metaName.indexOf('Data')>-1)) {
            var node = new ModelItem(this.$list, desc);
            this.nodes[desc.id] = node;
            this.nodesTime[desc.createdAt] = desc.id;
            node.$delete.on('click', event => {
                this.onNodeDeleteClicked(desc.id);
                event.stopPropagation();
                event.preventDefault();
            });
            node.$download.on('click', event => event.stopPropagation());
            node.$el.on('click', event => {
                this.onNodeClick(desc.id);
                //event.stopPropagation();
                //event.preventDefault();
            });
            node.$name.on('dblclick', event => {
                const name = $(event.target);
                name.editInPlace({
                    css: {
                        'z-index': 1000
                    },
                    onChange: (oldVal, newVal) => {
                        if (newVal && newVal !== oldVal) {
                            this.onNameChange(desc.id, newVal);
                        }
                    }
                });
            });
            if (desc.hashnew)
            {
                this.onAddDataHash(desc.id, desc.hash);
            }
            
        }
        if (desc.last==1)
        {
            this.addToTable();
        }
    };

    ArtifactIndexWidget.prototype.addToTable = function () {
        var self = this;
        var keys = Object.keys(self.nodesTime);
        var keyids = Object.keys(self.nodes);
        keys = keys.sort().reverse();
        var k = 0;
        var key = '';
        var id = '';
        for(k=0; k!=keys.length; k+=1)
        {
            key = keys[k];
            id = self.nodesTime[key];
            if (keyids.indexOf(id)>=0)
            {
                self.$list.append(self.nodes[id].$el);
            }
        }

    };

    ArtifactIndexWidget.prototype.removeNode = function (gmeId) {
        var node = this.nodes[gmeId];
        if (node) {
            node.remove();

            delete this.nodes[gmeId];
        }
    };

    ArtifactIndexWidget.prototype.updateNode = function (desc) {
        //if (desc  && desc.parentId === this.currentNode) {
        if (desc &&  (desc.metaName.indexOf('Data')>-1)) {
            if (desc.hashnew)
            {
                this.onAddDataHash(desc.id, desc.hash);
            }
            
            if (this.nodes[desc.id])
            {
                this.nodes[desc.id].update(desc);
            }
        }
    };

    /* * * * * * * * Visualizer event handlers * * * * * * * */


    /* * * * * * * * Visualizer life cycle callbacks * * * * * * * */
    ArtifactIndexWidget.prototype.destroy = function () {
    };

    ArtifactIndexWidget.prototype.onActivate = function () {
    };

    ArtifactIndexWidget.prototype.onDeactivate = function () {
    };

    return ArtifactIndexWidget;
});
