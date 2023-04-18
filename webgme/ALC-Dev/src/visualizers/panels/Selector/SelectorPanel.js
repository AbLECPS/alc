/*globals define, _, WebGMEGlobal*/
/*jshint browser: true*/
/**
 * Generated by VisualizerGenerator 1.7.0 from webgme on Thu Jul 07 2016 11:24:16 GMT-0500 (Central Daylight Time).
 */

define(['js/PanelBase/PanelBaseWithHeader',
    'js/PanelManager/IActivePanel',
    'widgets/Selector/SelectorWidget',
    './SelectorControl'
], function (PanelBaseWithHeader,
             IActivePanel,
             SelectorWidget,
             SelectorControl) {
    'use strict';

    var SelectorPanel;

    SelectorPanel = function (layoutManager, params) {
        var options = {};
        //set properties from options
        options[PanelBaseWithHeader.OPTIONS.LOGGER_INSTANCE_NAME] = 'SelectorPanel';
        options[PanelBaseWithHeader.OPTIONS.FLOATING_TITLE] = true;

        //call parent's constructor
        PanelBaseWithHeader.apply(this, [options, layoutManager]);

        this._client = params.client;

        this._config =  {
            navigationTitle: {
                enabled: true,
                attribute: 'name',
                depth: 20
            },
            byProjectKind: {
                navigationTitle: {}
            },
            byProjectId: {
                navigationTitle: {}
            }
        };

        //initialize UI
        this._initialize();

        this.logger.debug('ctor finished');
    };

    //inherit from PanelBaseWithHeader
    _.extend(SelectorPanel.prototype, PanelBaseWithHeader.prototype);
    _.extend(SelectorPanel.prototype, IActivePanel.prototype);

    SelectorPanel.prototype._initialize = function () {
        var self = this;

        //set Widget title
        this.setTitle('');

        this.widget = new SelectorWidget(this.logger, this.$el);

        this.widget.setTitle = function (title) {
            self.setTitle(title);
        };

        this.control = new SelectorControl({
            logger: this.logger,
            client: this._client,
            widget: this.widget
        });

        this.onActivate();
    };

    /* OVERRIDE FROM WIDGET-WITH-HEADER */
    /* METHOD CALLED WHEN THE WIDGET'S READ-ONLY PROPERTY CHANGES */
    SelectorPanel.prototype.onReadOnlyChanged = function (isReadOnly) {
        //apply parent's onReadOnlyChanged
        PanelBaseWithHeader.prototype.onReadOnlyChanged.call(this, isReadOnly);

    };

    SelectorPanel.prototype.onResize = function (width, height) {
        this.logger.debug('onResize --> width: ' + width + ', height: ' + height);
        this.widget.onWidgetContainerResize(width, height);
    };

    /* * * * * * * * Visualizer life cycle callbacks * * * * * * * */
    SelectorPanel.prototype.destroy = function () {
        this.control.destroy();
        this.widget.destroy();

        PanelBaseWithHeader.prototype.destroy.call(this);
        WebGMEGlobal.KeyboardManager.setListener(undefined);
        WebGMEGlobal.Toolbar.refresh();
    };

    SelectorPanel.prototype.onActivate = function () {
        this.widget.onActivate();
        this.control.onActivate();
        WebGMEGlobal.KeyboardManager.setListener(this.widget);
        WebGMEGlobal.Toolbar.refresh();
    };

    SelectorPanel.prototype.onDeactivate = function () {
        this.widget.onDeactivate();
        this.control.onDeactivate();
        WebGMEGlobal.KeyboardManager.setListener(undefined);
        WebGMEGlobal.Toolbar.refresh();
    };
   
    SelectorPanel.prototype.getValidTypesInfo = function (/*nodeId, aspect*/) {
        return {};
    };

    return SelectorPanel;
});
