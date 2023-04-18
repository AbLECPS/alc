define(['js/logger',
'jquery-fancytree/jquery.fancytree',
'jquery-fancytree/jquery.fancytree.table',
'jquery-fancytree/jquery.fancytree.gridnav',
'css!./styles/ActivityDefinitionTree.css'
], function (Logger) {
'use strict';

   function ActivityCampaignDefinitionTree(container, options) {

       this._logger = Logger.create('gme:Widgets:ActivityDefinition:ActivityCampaignDefinitionTree',
           WebGMEGlobal.gmeConfig.client.log);

       options = options || {};

       this._client = WebGMEGlobal.Client;

       this._el = container;

       this._initialize(options);

       this._logger.debug('Ctor finished...');
   }

   ActivityCampaignDefinitionTree.prototype._initialize = function (options) {
       var self = this;
       this._el.html('');

       let table = '<table id="campaign-tree"> \
           <colgroup> \
           <col width="5%" /> \
           <col width="50%"/> \
           </colgroup> \
           <thead> \
           <tr> \
               <th style="text-align:center">#</th> \
               <th style="text-align:center">Parameter</th> \
           </tr> \
           </thead> \
       </table>';

       this._treeEl = $(table, {});

       this._treeEl.fancytree({
           titlesTabbable: true,
           extensions: ["table", "gridnav"],
           table: {
             indentation: 20,
             nodeColumnIdx: 1
           },
           gridnav: {
             autofocusInput: false,
             handleCursorKeys: true,
           },
           click: function(event, data) {

           },
           dblclick: function(event, data)
           {

           },
           lazyLoad: function (event, data) {
               self._logger.debug('onLazyRead node:' + data.node.key);
               self.onNodeOpen.call(self, data.node.key);
               event.preventDefault();
               event.stopPropagation();
           },
           createNode: function(event, data) {
             var node = data.node,
               $tdList = $(node.tr).find(">td");

             if (node.isFolder()) {
                $tdList
                  .eq(1)
                  .prop("colspan", 5)
                  .nextAll()
                  .remove();
             }
           },
           renderColumns: function(event, data) {
             var node = data.node,
               $tdList = $(node.tr).find(">td");

             $tdList.eq(0).text(node.getIndexHier());
           },
           modifyChild: function(event, data) {
               data.tree.info(event.type, data);
           },
         })
         .on("nodeCommand", function(event, data) {
           var refNode,
             moveMode,
             tree = $.ui.fancytree.getTree(this),
             node = tree.getActiveNode();

           switch (data.cmd) {
             case "addChild":
             case "addSibling":
             case "indent":
             case "moveDown":
             case "moveUp":
             case "outdent":
             case "remove":
             case "rename":
             case "cut":
             case "copy":
             case "clear":
             case "paste":
               break;
             default:
               alert("Unhandled command: " + data.cmd);
               return;
           }
       });

       this._treeInstance = this._treeEl.fancytree('getTree');
       this._el.append(this._treeEl);
   };

   ActivityCampaignDefinitionTree.prototype.createNode = function (parentNode, objDescriptor) {
       var newNode;
       objDescriptor.name = objDescriptor.name || '';

       if (parentNode === null) 
       {
           parentNode = this._treeInstance.getRootNode();
       }

       newNode = parentNode.addChildren({
           title: objDescriptor.name,
           tooltip: "Id: " + objDescriptor.id + ", Name: " + objDescriptor.name,
           key: objDescriptor.id,
           folder: objDescriptor.folder,
           lazy: objDescriptor.hasChildren,
           extraClasses: objDescriptor.class || '',
           icon: objDescriptor.icon || null,
           isConnection: objDescriptor.isConnection,
           isAbstract: objDescriptor.isAbstract,
           isLibrary: objDescriptor.isLibrary,
           isLibraryRoot: objDescriptor.isLibraryRoot,
           libraryInfo: objDescriptor.libraryInfo,
           metaType: objDescriptor.metaType,
           isMetaNode: objDescriptor.isMetaNode, 
           data: objDescriptor.data
       });

       this._logger.debug('New node created.');

       return newNode;
   };

   ActivityCampaignDefinitionTree.prototype.onNodeOpen = function (nodeId) {
       this._logger.warn('Default onNodeOpen for node ' +
           nodeId + ' called, doing nothing. Please override onNodeOpen(nodeId)');
   };

return ActivityCampaignDefinitionTree;
});