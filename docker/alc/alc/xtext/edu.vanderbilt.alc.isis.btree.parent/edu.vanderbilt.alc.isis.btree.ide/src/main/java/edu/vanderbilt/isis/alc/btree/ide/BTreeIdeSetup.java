/*
 * generated by Xtext 2.25.0
 */
package edu.vanderbilt.isis.alc.btree.ide;

import com.google.inject.Guice;
import com.google.inject.Injector;
import edu.vanderbilt.isis.alc.btree.BTreeRuntimeModule;
import edu.vanderbilt.isis.alc.btree.BTreeStandaloneSetup;
import org.eclipse.xtext.util.Modules2;

/**
 * Initialization support for running Xtext languages as language servers.
 */
public class BTreeIdeSetup extends BTreeStandaloneSetup {

	@Override
	public Injector createInjector() {
		return Guice.createInjector(Modules2.mixin(new BTreeRuntimeModule(), new BTreeIdeModule()));
	}
	
}
