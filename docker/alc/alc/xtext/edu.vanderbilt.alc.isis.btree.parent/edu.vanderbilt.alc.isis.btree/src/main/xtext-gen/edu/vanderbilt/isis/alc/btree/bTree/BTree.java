/**
 * generated by Xtext 2.25.0
 */
package edu.vanderbilt.isis.alc.btree.bTree;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>BTree</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.BTree#getBtree <em>Btree</em>}</li>
 * </ul>
 *
 * @see edu.vanderbilt.isis.alc.btree.bTree.BTreePackage#getBTree()
 * @model
 * @generated
 */
public interface BTree extends EObject
{
  /**
   * Returns the value of the '<em><b>Btree</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the value of the '<em>Btree</em>' containment reference.
   * @see #setBtree(BTreeNode)
   * @see edu.vanderbilt.isis.alc.btree.bTree.BTreePackage#getBTree_Btree()
   * @model containment="true"
   * @generated
   */
  BTreeNode getBtree();

  /**
   * Sets the value of the '{@link edu.vanderbilt.isis.alc.btree.bTree.BTree#getBtree <em>Btree</em>}' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Btree</em>' containment reference.
   * @see #getBtree()
   * @generated
   */
  void setBtree(BTreeNode value);

} // BTree
