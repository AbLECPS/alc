define(['js/logger',
'jquery-fancytree/jquery.fancytree',
'jquery-fancytree/jquery.fancytree.table',
'jquery-fancytree/jquery.fancytree.gridnav',
'css!./styles/ActivityDefinitionTree.css'
], function (Logger) {
'use strict';

   function ActivityInputDefinitionTree(container, options) {

       this._logger = Logger.create('gme:Widgets:ActivityDefinition:ActivityInputDefinitionTree',
           WebGMEGlobal.gmeConfig.client.log);

       options = options || {};

       this._client = WebGMEGlobal.Client;

       this._client.startTransaction();

       this._el = container;

       this._initialize(options);

       this._logger.debug('Ctor finished...');
   }

   ActivityInputDefinitionTree.prototype._initialize = function (options) {
       var self = this;
       this._el.html('');

       let table = '<table id="input-tree"> \
           <colgroup> \
           <col width="5%" /> \
           <col width="50%"/> \
           </colgroup> \
           <thead> \
           <tr> \
               <th>#</th> \
               <th>Dataset</th> \
           </tr> \
           </thead> \
       </table>';

       this._treeEl = $(table, {});

       this._treeEl.fancytree({
           selectMode: 3,
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
           init: function(event, data) {

           },
           click: function(event, data) {

           },
           dblclick: function(event, data)
           {
            
           },
           select: function(event, data) {      
            if(data.node.parent.parent.data.multiselect === false)
            {
              self.clearInput(data.node.parent.parent.data.id);
              self.addInput(data.node.parent.parent.data.id, data.node.data.id);
            }
            else
            {
              self.clearInput(data.node.parent.data.id);
              var children = data.tree.getSelectedNodes();
              for(var d of children)
              {
                if(d.data.id)
                {
                  self.addInput(data.node.parent.data.id, d.data.id);
                }
              }
            }
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

            $tdList.eq(0).css("background-color","white");

            $tdList.eq(3).css("display", "flex");

            $tdList.eq(3).css("align-items", "center");

            $tdList.eq(3).css("justify-content", "center");

            if (node.isFolder()) 
            {
                $tdList
                  .eq(1)
                  .prop("colspan", 5)
                  .nextAll()
                  .remove();
            }
            else
            {
              var rootNode = data.node.parent.parent;

              if(rootNode)
              {
                var n = self._client.getNode(rootNode.data.id);
                var ids = n.getMemberIds('Data');
                if(ids.includes(data.node.data.id))
                {
                  data.node.setSelected(true);
                }
              }

              var selNodes = data.node.parent.getSelectedNodes();
              if(selNodes.length === data.node.parent.children.length)
              {
                data.node.parent.setSelected(true);
              }
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

   ActivityInputDefinitionTree.prototype.clearInput = function (parentId)
   {
      if(parentId === undefined)
        return;
      var n = this._client.getNode(parentId);
      var ids = n.getMemberIds('Data');
      for(var i = 0;i < ids.length;++i)
      {
          this._client.removeMember(parentId, ids[i], "Data");
      }
   };

   ActivityInputDefinitionTree.prototype.addInput = function (parentId, childId)
   {  
      if(parentId === undefined || childId === undefined)
        return;
      this._client.addMember(parentId, childId, "Data");
   };

   ActivityInputDefinitionTree.prototype.createNode = function (parentNode, objDescriptor) {
       var newNode;
       objDescriptor.name = objDescriptor.name || '';

       if (parentNode === null) 
       {
           parentNode = this._treeInstance.getRootNode();
       }

       newNode = parentNode.addChildren({
           title: objDescriptor.name,
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
           data: objDescriptor.data,
           checkbox: objDescriptor.checkbox,
           radiogroup: objDescriptor.radiogroup
       });

       this._logger.debug('New node created.');

       return newNode;
   };

   ActivityInputDefinitionTree.prototype.onNodeOpen = function (nodeId) {
       this._logger.warn('Default onNodeOpen for node ' +
           nodeId + ' called, doing nothing. Please override onNodeOpen(nodeId)');
   };

return ActivityInputDefinitionTree;
});