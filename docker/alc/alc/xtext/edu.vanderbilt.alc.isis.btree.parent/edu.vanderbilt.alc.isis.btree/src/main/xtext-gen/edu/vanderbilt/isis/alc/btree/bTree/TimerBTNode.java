/**
 * generated by Xtext 2.25.0
 */
package edu.vanderbilt.isis.alc.btree.bTree;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Timer BT Node</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.TimerBTNode#getName <em>Name</em>}</li>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.TimerBTNode#getDuration <em>Duration</em>}</li>
 * </ul>
 *
 * @see edu.vanderbilt.isis.alc.btree.bTree.BTreePackage#getTimerBTNode()
 * @model
 * @generated
 */
public interface TimerBTNode extends BTreeNode
{
  /**
   * Returns the value of the '<em><b>Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the value of the '<em>Name</em>' attribute.
   * @see #setName(String)
   * @see edu.vanderbilt.isis.alc.btree.bTree.BTreePackage#getTimerBTNode_Name()
   * @model
   * @generated
   */
  String getName();

  /**
   * Sets the value of the '{@link edu.vanderbilt.isis.alc.btree.bTree.TimerBTNode#getName <em>Name</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Name</em>' attribute.
   * @see #getName()
   * @generated
   */
  void setName(String value);

  /**
   * Returns the value of the '<em><b>Duration</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the value of the '<em>Duration</em>' attribute.
   * @see #setDuration(String)
   * @see edu.vanderbilt.isis.alc.btree.bTree.BTreePackage#getTimerBTNode_Duration()
   * @model
   * @generated
   */
  String getDuration();

  /**
   * Sets the value of the '{@link edu.vanderbilt.isis.alc.btree.bTree.TimerBTNode#getDuration <em>Duration</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Duration</em>' attribute.
   * @see #getDuration()
   * @generated
   */
  void setDuration(String value);

} // TimerBTNode