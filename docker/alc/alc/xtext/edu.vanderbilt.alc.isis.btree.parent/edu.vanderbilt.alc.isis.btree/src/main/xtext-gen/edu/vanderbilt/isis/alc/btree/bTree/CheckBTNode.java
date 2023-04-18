/**
 * generated by Xtext 2.25.0
 */
package edu.vanderbilt.isis.alc.btree.bTree;

import org.eclipse.emf.common.util.EList;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Check BT Node</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.CheckBTNode#getCheck <em>Check</em>}</li>
 * </ul>
 *
 * @see edu.vanderbilt.isis.alc.btree.bTree.BTreePackage#getCheckBTNode()
 * @model
 * @generated
 */
public interface CheckBTNode extends BTreeNode
{
  /**
   * Returns the value of the '<em><b>Check</b></em>' reference list.
   * The list contents are of type {@link edu.vanderbilt.isis.alc.btree.bTree.CheckNode}.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the value of the '<em>Check</em>' reference list.
   * @see edu.vanderbilt.isis.alc.btree.bTree.BTreePackage#getCheckBTNode_Check()
   * @model
   * @generated
   */
  EList<CheckNode> getCheck();

} // CheckBTNode