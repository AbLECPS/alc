/*globals define, */
define([
    'js/Layouts/DefaultLayout/DefaultLayout',
    'text!./templates/DefFloatLayout.html'
], function(
    DefaultLayout,
    DefFloatTemplate
) {
    'use strict';
    
    var DefFloatLayout = function(params) {
        params = params || {};
        params.template = DefFloatTemplate;
        DefaultLayout.call(this, params);
        this.newconfig = WebGMEGlobal.componentSettings[this.getComponentId()];
        this.panels.push(this.newconfig.panels[0]); 
        
    };

    DefFloatLayout.prototype = Object.create(DefaultLayout.prototype);

    DefFloatLayout.prototype.getComponentId = function () {
        return 'DefFloatLayout';
    };

    /**
     * Initialize the html page. This example is using the jQuery Layout plugin.
     *
     * @return {undefined}
     */
    DefFloatLayout.prototype.init = function() {
        DefaultLayout.prototype.init.apply(this, arguments);
        this._floatPanel = this._body.find('div.float');
        this._centerPanel = this._body.find('div.layout-center');
    };

    /**
     * Add a panel to a given container. This is defined in the corresponding
     * layout config JSON file.
     *
     * @param {Panel} panel
     * @param {String} container
     * @return {undefined}
     */
    DefFloatLayout.prototype.addToContainer = function(panel, container) {
        if (container === 'float') {
            this._floatPanel.append(panel.$pEl);
        } else {
            DefaultLayout.prototype.addToContainer.apply(this, arguments);
        }
    };

    // DefFloatLayout.prototype._onCenterResize = function() {
        // var width = this._centerPanel.width() - this._sidebarPanel.width();
        // this._canvas.setSize(width, this._centerPanel.height());
    // };

    return DefFloatLayout;
});
