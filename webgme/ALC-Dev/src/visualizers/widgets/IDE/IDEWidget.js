/*globals define, WebGMEGlobal*/
/*jshint browser: true*/

/**
 * Generated by VisualizerGenerator 1.7.0 from webgme on Thu Jul 07 2016 11:24:16 GMT-0500 (Central Daylight Time).
 */

define(['css!./styles/IDEWidget.css',
        '../library/bootstrap3-editable-1.5.1/bootstrap3-editable/js/bootstrap-editable.min',
        'css!../library/bootstrap3-editable-1.5.1/bootstrap3-editable/css/bootstrap-editable.css'],
    function () {
        'use strict';

        var IDEWidget,
            WIDGET_CLASS = 'm-viz';

        IDEWidget = function (logger, container) {
            this._logger = logger.fork('Widget');

            this._el = container;

            this.nodes = {};
            this._initialize();
            this.url = '';
            this.table_rendered = 0;
            this._logger.debug('ctor finished');
        };

        IDEWidget.prototype._initialize = function () {
            var width = this._el.width(),
                height = this._el.height(),
                self = this;

            $.fn.editable.defaults.mode = 'inline';
            
            // set widget class
            this._el.addClass(WIDGET_CLASS);

            var dummy = document.createElement('div');
            dummy.id = 'dummyheader';
            dummy.setAttribute("style", "margin-top: 50px;margin-left: 50px");
            this._el.append(dummy);

        };

        IDEWidget.prototype.update_url = function (url) {
            var self = this;
            if (self.url != url)
            {
                self.url = url;
                if (self.url == '' || self.url == '{}')
                {
                    document.getElementById("VSCode").disabled = true;
                    document.getElementById("VNC").disabled = true;
                    document.getElementById("VSCode").innerHTML = "VSCode";
                    document.getElementById("VNC").innerHTML = "VNC";
                }
                else {
                    var h = window.location.hostname;
                    var p = window.location.port;
                    var csurl = '';
                    var vncurl = '';
                    if (p)
                    {
                        csurl = 'http://'+h + ":" + p +'/cs/'+ self.url+'/';
                        vncurl = 'http://'+h+ ":" + p+'/vnc/'+self.url+'/vnc_auto.html?path=/vnc/'+self.url+'/&password=vncpassword';
                    }
                    else{
                        csurl = 'https://'+h + '/cs/'+ self.url+'/';
                        vncurl = 'https://'+h+'/vnc/'+self.url+'/vnc_auto.html?path=/vnc/'+self.url+'/&password=vncpassword';
                    }
                    document.getElementById("VSCode").disabled = false;
                    document.getElementById("VNC").disabled = false;
                    
                    document.getElementById("VSCode").innerHTML = '<a href="'+csurl+'" target="_blank"> VSCode </a>';
                    document.getElementById("VNC").innerHTML = '<a href="'+vncurl+'" target="_blank"> VNC </a>';
                }

            }
            

            

        };


        IDEWidget.prototype.renderTable = function () {
            var self = this;

            var table = document.createElement('div');
            table.className = 'm-viz-table'
            table.id = 'dep-table';
            table.setAttribute("style", "margin-top: 50px;margin-left: 50px");

            var node = document.createElement('div');
            node.className = 'm-viz-heading';

            var cnode = document.createElement('div');
            cnode.className = 'm-viz-table-col';
            var text = "Links";
            var chld1 = document.createTextNode(text);
            cnode.appendChild(chld1);
            node.appendChild(cnode);
            table.appendChild(node);
            self._el.append(table);
            self.table = this._el.find('#dep-table');

            node = document.createElement('div');
            node.className = 'm-viz-table-row';
            node.setAttribute("style", "background-color:#eee");

            var chld = document.createElement('div');
            chld.className = 'm-viz-table-col-wide-mod';
            chld.id = 'VSCode';
            chld.id2 = 0;
            var mtext = 'VSCode'
            chld.innerHTML = mtext;
            node.appendChild(chld);
            table.appendChild(node);
            self._el.append(table);

            node = document.createElement('div');
            node.className = 'm-viz-table-row';
            node.setAttribute("style", "background-color:#eee");
            
            chld = document.createElement('div');
            chld.className = 'm-viz-table-col-wide-mod';
            chld.id = 'VNC';
            chld.id2 = 0;
            var mtext = 'VNC'
            chld.innerHTML = mtext;
            node.appendChild(chld);
            table.appendChild(node);
            self._el.append(table);


            if (self.url == '' || self.url == '{}')
            {
                document.getElementById("VSCode").disabled = true;
                document.getElementById("VNC").disabled = true;
            }
            
            self.table_rendered = 1;


        };

        


        IDEWidget.prototype.onWidgetContainerResize = function (width, height) {
            // this._logger.debug('Widget is resizing...');
        };

        // Adding/Removing/Updating items
        IDEWidget.prototype.addNode = function (desc) {
            var self = this;
            if (desc) {

                if (self.table_rendered == 0) {
                    self.renderTable();
                }

                self.update_url(desc.url);
                

            }

        };

        IDEWidget.prototype.removeNode = function (gmeId) {
            var desc = this.nodes[gmeId];
            if (desc) {
                //this._el.append('<div>Removing node "' + desc.name + '"</div>');
                delete this.nodes[gmeId];
            }
        };

        IDEWidget.prototype.updateNode = function (desc) {
            var self = this;
            if (desc) {
                if (self.table_rendered == 0) {
                    self.renderTable();
                }
                self.update_url(desc.url);
                //this._el.append('<div>Updating node "' + desc.name + '"</div>');
            }
        };

        /* * * * * * * * Visualizer event handlers * * * * * * * */

        IDEWidget.prototype.onNodeClick = function (/*id*/) {
            // This currently changes the active node to the given id and
            // this is overridden in the controller.
        };

        IDEWidget.prototype.onEditModeInfo = function (/*id*/) {
            // This currently changes the active node to the given id and
            // this is overridden in the controller.
        }

        IDEWidget.prototype.onBackgroundDblClick = function () {

        };

        /* * * * * * * * Visualizer life cycle callbacks * * * * * * * */
        IDEWidget.prototype.destroy = function () {
        };

        IDEWidget.prototype.onActivate = function () {
            this._logger.debug('IDEWidget has been activated');
        };

        IDEWidget.prototype.onDeactivate = function () {
            this._logger.debug('IDEWidget has been deactivated');
        };

        return IDEWidget;
    });