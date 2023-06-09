'use strict';

import * as path from 'path';
import * as os from 'os';

import {Trace} from 'vscode-jsonrpc';
import { commands, window, workspace, ExtensionContext, Uri } from 'vscode';
import { LanguageClient, LanguageClientOptions, ServerOptions } from 'vscode-languageclient';

export function activate(context: ExtensionContext) {
    // The server is a locally installed in src/btree
    let launcher = os.platform() === 'win32' ? 'btree-standalone.bat' : 'btree-standalone';
    let script = context.asAbsolutePath(path.join('src', 'btree', 'bin', launcher));

    let serverOptions: ServerOptions = {
        run : { command: script },
        debug: { command: script, args: [], options: { env: createDebugEnv() } }
    };
    
    let clientOptions: LanguageClientOptions = {
        documentSelector: ['btree'],
        outputChannelName: 'BTree',
        synchronize: {
            fileEvents: workspace.createFileSystemWatcher('**/*.bt')
        }
    };
    
    // Create the language client and start the client.
    let lc = new LanguageClient('btree', serverOptions, clientOptions);
    let orange = window.createOutputChannel("Btree-Command");
    
    
    // enable tracing (.Off, .Messages, Verbose)
    lc.trace = Trace.Verbose;
    let disposable = lc.start();
    
    // Push the disposable to the context's subscriptions so that the 
    // client can be deactivated on extension deactivation
    context.subscriptions.push(disposable);
}

function createDebugEnv() {
    return Object.assign({
        JAVA_OPTS:"-Xdebug -Xrunjdwp:server=y,transport=dt_socket,address=8000,suspend=n,quiet=y"
    }, process.env)
}